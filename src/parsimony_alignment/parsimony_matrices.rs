use crate::alignment::{Alignment, Mapping};
use crate::cmp_f64;
use crate::parsimony_alignment::{
    parsimony_info::ParsimonySiteInfo,
    Direction::{self, GapX, GapY, Matc},
};
use std::f64::INFINITY as INF;
use std::fmt;

use super::parsimony_costs::BranchParsimonyCosts;
use super::parsimony_info::GapFlag::{GapExt, GapFixed, GapOpen, NoGap};
use super::parsimony_sets::{gap_set, ParsimonySet};

pub(super) struct ScoreMatrices {
    pub(super) m: Vec<Vec<f64>>,
    pub(super) x: Vec<Vec<f64>>,
    pub(super) y: Vec<Vec<f64>>,
}

impl ScoreMatrices {
    pub(super) fn new(len1: usize, len2: usize) -> ScoreMatrices {
        ScoreMatrices {
            m: vec![vec![0.0; len2]; len1],
            x: vec![vec![0.0; len2]; len1],
            y: vec![vec![0.0; len2]; len1],
        }
    }
}

pub(super) struct TracebackMatrices {
    pub(super) m: Vec<Vec<Direction>>,
    pub(super) x: Vec<Vec<Direction>>,
    pub(super) y: Vec<Vec<Direction>>,
}

impl TracebackMatrices {
    pub(super) fn new(len1: usize, len2: usize) -> TracebackMatrices {
        TracebackMatrices {
            m: vec![vec![Matc; len2]; len1],
            x: vec![vec![GapX; len2]; len1],
            y: vec![vec![GapY; len2]; len1],
        }
    }
}

pub(crate) struct ParsimonyAlignmentMatrices {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(super) score: ScoreMatrices,
    pub(super) trace: TracebackMatrices,
    pub(super) direction_picker: [&'static [Direction]; 8],
    pub(crate) rng: fn(usize) -> usize,
}

impl fmt::Display for ParsimonyAlignmentMatrices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "S.M").unwrap();
        for row in &self.score.m {
            writeln!(f, "{:?}", row).unwrap();
        }
        writeln!(f, "S.X").unwrap();
        for row in &self.score.x {
            writeln!(f, "{:?}", row).unwrap();
        }
        writeln!(f, "S.Y").unwrap();
        for row in &self.score.y {
            writeln!(f, "{:?}", row).unwrap();
        }
        writeln!(f, "T.M").unwrap();
        for row in &self.trace.m {
            writeln!(f, "{:?}", row).unwrap();
        }
        writeln!(f, "T.X").unwrap();
        for row in &self.trace.x {
            writeln!(f, "{:?}", row).unwrap();
        }
        writeln!(f, "T.Y").unwrap();
        for row in &self.trace.y {
            writeln!(f, "{:?}", row).unwrap();
        }
        Ok(())
    }
}

impl ParsimonyAlignmentMatrices {
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        rng: fn(usize) -> usize,
    ) -> ParsimonyAlignmentMatrices {
        ParsimonyAlignmentMatrices {
            rows,
            cols,
            score: ScoreMatrices::new(rows, cols),
            trace: TracebackMatrices::new(rows, cols),
            direction_picker: [
                /* 000 */ &[][..],
                /* 001 */ &[Matc][..],
                /* 010 */ &[GapX][..],
                /* 011 */ &[Matc, GapX][..],
                /* 100 */ &[GapY][..],
                /* 101 */ &[Matc, GapY][..],
                /* 110 */ &[GapX, GapY][..],
                /* 111 */ &[Matc, GapY, GapX][..],
            ],
            rng,
        }
    }

    pub(crate) fn fill_matrices(
        &mut self,
        l_info: &[ParsimonySiteInfo],
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_info: &[ParsimonySiteInfo],
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) {
        self.init_x(l_info, l_scoring);
        self.init_y(r_info, r_scoring);
        for i in 1..self.rows {
            for j in 1..self.cols {
                if l_info[i - 1].is_fixed() || r_info[j - 1].is_fixed() {
                    let ni = i - l_info[i - 1].is_fixed() as usize;
                    let nj = j - r_info[j - 1].is_fixed() as usize;
                    self.score.m[i][j] = self.score.m[ni][nj];
                    self.score.x[i][j] = self.score.x[ni][nj];
                    self.score.y[i][j] = self.score.y[ni][nj];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) =
                        self.possible_match(i - 1, j - 1, l_info, l_scoring, r_info, r_scoring);
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.possible_gap_x(i - 1, j, l_info, l_scoring, r_info);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, l_info, r_info, r_scoring);
                }
            }
        }
    }

    pub(crate) fn traceback(
        &self,
        l_info: &[ParsimonySiteInfo],
        r_info: &[ParsimonySiteInfo],
    ) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
        let mut i = self.rows - 1;
        let mut j = self.cols - 1;
        let (pars_score, mut action) =
            self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]);
        let max_alignment_length = l_info.len() + r_info.len();
        let mut node_info = Vec::<ParsimonySiteInfo>::with_capacity(max_alignment_length);
        let mut alignment = Alignment::new(
            Mapping::with_capacity(max_alignment_length),
            Mapping::with_capacity(max_alignment_length),
        );
        while i > 0 || j > 0 {
            if (i > 0 && l_info[i - 1].is_fixed()) || (j > 0 && r_info[j - 1].is_fixed()) {
                if i > 0 && l_info[i - 1].is_fixed() {
                    alignment.map_x.push(Some(i - 1));
                    alignment.map_y.push(None);
                    node_info.push(ParsimonySiteInfo::new(gap_set(), GapFixed));
                    i -= 1;
                }
                if j > 0 && r_info[j - 1].is_fixed() {
                    alignment.map_x.push(None);
                    alignment.map_y.push(Some(j - 1));
                    node_info.push(ParsimonySiteInfo::new(gap_set(), GapFixed));
                    j -= 1;
                }
            } else {
                match action {
                    Matc => {
                        assert!(!l_info[i - 1].is_fixed() && !r_info[j - 1].is_fixed());
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(Some(j - 1));
                        let mut set = &l_info[i - 1].set & &r_info[j - 1].set;
                        if set.is_empty() {
                            set = &l_info[i - 1].set | &r_info[j - 1].set;
                        }
                        node_info.push(ParsimonySiteInfo::new(set, NoGap));
                        action = self.trace.m[i][j];
                        i -= 1;
                        j -= 1;
                    }
                    GapX => {
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(None);
                        match l_info[i - 1].flag {
                            GapFixed => unreachable!(),
                            GapOpen | GapExt => {
                                node_info.push(ParsimonySiteInfo::new(gap_set(), GapFixed))
                            }
                            NoGap => {
                                node_info.push(ParsimonySiteInfo::new(
                                    l_info[i - 1].set.clone(),
                                    if self.trace.x[i][j] != GapX
                                        || (i > 1
                                            && self.trace.x[i][j] == GapX
                                            && l_info[i - 2].is_possible()
                                            && self.trace.x[i - 1][j] != GapY)
                                        || i == 1
                                    {
                                        GapOpen
                                    } else {
                                        GapExt
                                    },
                                ));
                            }
                        }
                        action = self.trace.x[i][j];
                        i -= 1;
                    }
                    GapY => {
                        alignment.map_x.push(None);
                        alignment.map_y.push(Some(j - 1));
                        match r_info[j - 1].flag {
                            GapFixed => unreachable!(),
                            GapOpen | GapExt => {
                                node_info.push(ParsimonySiteInfo::new(gap_set(), GapFixed))
                            }
                            NoGap => {
                                node_info.push(ParsimonySiteInfo::new(
                                    r_info[j - 1].set.clone(),
                                    if self.trace.y[i][j] != GapY
                                        || (j > 1
                                            && self.trace.y[i][j] == GapY
                                            && r_info[j - 2].is_possible()
                                            && self.trace.y[i][j - 1] != GapX)
                                        || j == 1
                                    {
                                        GapOpen
                                    } else {
                                        GapExt
                                    },
                                ));
                            }
                        }
                        action = self.trace.y[i][j];
                        j -= 1;
                    }
                }
            }
        }
        node_info.reverse();
        alignment.map_x.reverse();
        alignment.map_y.reverse();
        (node_info, alignment, pars_score)
    }

    fn init_x(
        &mut self,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<&dyn BranchParsimonyCosts>,
    ) {
        for i in 1..self.rows {
            self.score.x[i][0] = self.score.x[i - 1][0];
            if node_info[i - 1].flag == NoGap {
                if self.score.x[i - 1][0] == 0.0 {
                    self.score.x[i][0] += scoring.gap_open_cost();
                } else {
                    self.score.x[i][0] += scoring.gap_ext_cost();
                }
            } 
            if node_info[i - 1].flag != GapFixed {
                self.trace.m[i][0] = GapX;
                self.trace.x[i][0] = GapX;
                self.trace.y[i][0] = GapX;
            }
            self.score.y[i][0] = INF;
            self.score.m[i][0] = INF;
        }
    }

    fn init_y(
        &mut self,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<&dyn BranchParsimonyCosts>,
    ) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1];
            if node_info[j - 1].flag == NoGap {
                if self.score.y[0][j - 1] == 0.0 {
                    self.score.y[0][j] += scoring.gap_open_cost();
                } else {
                    self.score.y[0][j] += scoring.gap_ext_cost();
                }
            }
            if node_info[j - 1].flag != GapFixed {
                self.trace.m[0][j] = GapY;
                self.trace.x[0][j] = GapY;
                self.trace.y[0][j] = GapY;
            }
            self.score.x[0][j] = INF;
            self.score.m[0][j] = INF;
        }
    }

    fn possible_gap_x(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_info: &[ParsimonySiteInfo],
    ) -> (f64, Direction) {
        match l_info[i].flag {
            GapOpen | GapFixed => {
                self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j])
            }
            GapExt => {
                let m_gap_adjustment = l_scoring.gap_open_cost() - l_scoring.gap_ext_cost();
                let y_gap_adjustment =
                    self.left_gap_cost_adjustment(i, j, l_info, l_scoring, r_info);
                self.select_direction(
                    self.score.m[i][j] + m_gap_adjustment,
                    self.score.x[i][j],
                    self.score.y[i][j] + y_gap_adjustment,
                )
            }
            NoGap => self.select_direction(
                self.score.m[i][j] + l_scoring.gap_open_cost(),
                self.gap_x_score(i, j, l_info, l_scoring),
                self.score.y[i][j] + l_scoring.gap_open_cost(),
            ),
        }
    }

    fn left_gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_info: &[ParsimonySiteInfo],
    ) -> f64 {
        let mut y_gap_adjustment = 0.0;
        let mut skip_i = i;
        let mut skip_j = j;
        while skip_j > 0
            && skip_i > 0
            && (l_info[skip_i - 1].is_fixed()
                || r_info[skip_j - 1].is_fixed()
                || self.trace.y[skip_i][skip_j] == GapY)
        {
            if l_info[skip_i - 1].is_fixed() {
                skip_i -= 1;
            }
            if r_info[skip_j - 1].is_fixed() || self.trace.y[skip_i][skip_j] == GapY {
                skip_j -= 1;
            }
        }
        if self.trace.y[skip_i][skip_j] != GapX {
            y_gap_adjustment = l_scoring.gap_open_cost() - l_scoring.gap_ext_cost();
        }
        y_gap_adjustment
    }

    fn possible_gap_y(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        r_info: &[ParsimonySiteInfo],
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        match r_info[j].flag {
            GapFixed | GapOpen => {
                self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j])
            }
            GapExt => {
                let x_gap_adjustment =
                    self.right_gap_cost_adjustment(i, j, l_info, r_info, r_scoring);
                let m_gap_adjustment = r_scoring.gap_open_cost() - r_scoring.gap_ext_cost();
                self.select_direction(
                    self.score.m[i][j] + m_gap_adjustment,
                    self.score.x[i][j] + x_gap_adjustment,
                    self.score.y[i][j],
                )
            }
            NoGap => self.select_direction(
                self.score.m[i][j] + r_scoring.gap_open_cost(),
                self.score.x[i][j] + r_scoring.gap_open_cost(),
                self.gap_y_score(i, j, r_info, r_scoring),
            ),
        }
    }

    fn right_gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        r_info: &[ParsimonySiteInfo],
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut x_gap_adjustment = 0.0;
        let mut skip_i = i;
        let mut skip_j = j;
        while skip_i > 0
            && skip_j > 0
            && (r_info[skip_j - 1].is_fixed()
                || l_info[skip_i - 1].is_fixed()
                || self.trace.x[skip_i][skip_j] == GapX)
        {
            if l_info[skip_i - 1].is_fixed() || self.trace.x[skip_i][skip_j] == GapX {
                skip_i -= 1;
            }
            if r_info[skip_j - 1].is_fixed() {
                skip_j -= 1;
            }
        }
        if self.trace.x[skip_i][skip_j] != GapY {
            x_gap_adjustment = r_scoring.gap_open_cost() - r_scoring.gap_ext_cost();
        }
        x_gap_adjustment
    }

    fn possible_match(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_info: &[ParsimonySiteInfo],
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        let (x_gap_adjustment, y_gap_adjustment) =
            self.gap_cost_adjustment(i, j, l_info, l_scoring, r_info, r_scoring);
        let score = self.get_match_cost(&l_info[i].set, l_scoring, &r_info[j].set, r_scoring);
        self.select_direction(
            self.score.m[i][j] + score,
            self.score.x[i][j] + x_gap_adjustment + score,
            self.score.y[i][j] + y_gap_adjustment + score,
        )
    }

    fn gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        l_info: &[ParsimonySiteInfo],
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_info: &[ParsimonySiteInfo],
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> (f64, f64) {
        let mut x_gap_adjustment = 0.0;
        let mut y_gap_adjustment = 0.0;
        if l_info[i].is_ext() || r_info[j].is_ext() {
            if l_info[i].is_ext() {
                let skip_j = (1..=j)
                    .rev()
                    .find(|&skip_j| {
                        self.trace.y[i][skip_j] != GapY && !r_info[skip_j - 1].is_fixed()
                    })
                    .unwrap_or(1);
                if self.trace.y[i][skip_j] != Matc {
                    y_gap_adjustment += l_scoring.gap_open_cost() - l_scoring.gap_ext_cost();
                }
            } else {
                y_gap_adjustment += l_scoring.gap_open_cost() - l_scoring.gap_ext_cost();
            }
            if r_info[j].is_ext() {
                let skip_i = (1..=i)
                    .rev()
                    .find(|&skip_i| {
                        self.trace.x[skip_i][j] != GapX && !l_info[skip_i - 1].is_fixed()
                    })
                    .unwrap_or(1);
                if self.trace.x[skip_i][j] != Matc {
                    x_gap_adjustment += r_scoring.gap_open_cost() - r_scoring.gap_ext_cost();
                }
            } else {
                x_gap_adjustment += r_scoring.gap_open_cost() - r_scoring.gap_ext_cost();
            }
        }
        (x_gap_adjustment, y_gap_adjustment)
    }

    fn get_match_cost(
        &self,
        l_set: &ParsimonySet,
        l_scoring: &Box<&dyn BranchParsimonyCosts>,
        r_set: &ParsimonySet,
        r_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        (l_set | r_set)
            .into_iter()
            .map(|a| min_score(l_set, l_scoring, a) + min_score(r_set, r_scoring, a))
            .min_by(cmp_f64())
            .unwrap_or(INF)
    }

    fn select_direction(&self, sm: f64, sx: f64, sy: f64) -> (f64, Direction) {
        let (mut min_val, mut sel_mat) = (sm, 0b001);
        if sx < min_val {
            (min_val, sel_mat) = (sx, 0b010);
        } else if sx == min_val {
            sel_mat |= 0b010;
        }
        if sy < min_val {
            (min_val, sel_mat) = (sy, 0b100);
        } else if sy == min_val {
            sel_mat |= 0b100;
        }
        (
            min_val,
            self.direction_picker[sel_mat][(self.rng)(self.direction_picker[sel_mat].len())],
        )
    }

    fn gap_x_score(
        &self,
        i: usize,
        j: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut skip_i = i;
        while skip_i > 0
            && (node_info[skip_i - 1].is_open() || node_info[skip_i - 1].is_ext())
            && self.trace.x[skip_i][j] == GapX
        {
            skip_i -= 1;
        }
        if self.trace.x[skip_i][j] != GapX
            && (skip_i == 0 || node_info[skip_i - 1].is_open() || node_info[skip_i - 1].is_ext())
        {
            self.score.x[skip_i][j] + scoring.gap_open_cost()
        } else {
            self.score.x[skip_i][j] + scoring.gap_ext_cost()
        }
    }

    fn gap_y_score(
        &self,
        i: usize,
        j: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut skip_j = j;
        while skip_j > 0 && node_info[skip_j - 1].is_open() && self.trace.y[i][skip_j] == GapY {
            skip_j -= 1;
        }
        if self.trace.y[i][skip_j] != GapY && (skip_j == 0 || node_info[skip_j - 1].is_open()) {
            self.score.y[i][skip_j] + scoring.gap_open_cost()
        } else {
            self.score.y[i][skip_j] + scoring.gap_ext_cost()
        }
    }
}

fn min_score(set: &ParsimonySet, scoring: &Box<&dyn BranchParsimonyCosts>, ancestor: u8) -> f64 {
    set.into_iter()
        .map(|l: &u8| scoring.match_cost(ancestor, *l))
        .min_by(cmp_f64())
        .unwrap()
}

#[cfg(test)]
mod parsimony_matrices_tests;
