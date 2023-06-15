use crate::alignment::{Alignment, Mapping};
use crate::cmp_f64;
use crate::parsimony_alignment::parsimony_sets;
use crate::parsimony_alignment::{
    parsimony_info::ParsimonySiteInfo,
    Direction::{self, GapX, GapY, Matc, Skip},
};
use std::f64::INFINITY as INF;
use std::fmt;

use super::parsimony_costs::BranchParsimonyCosts;
use super::parsimony_info::GapFlag::{GapExt, GapFixed, GapOpen, NoGap};
use super::parsimony_sets::ParsimonySet;

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
            m: vec![vec![Skip; len2]; len1],
            x: vec![vec![Skip; len2]; len1],
            y: vec![vec![Skip; len2]; len1],
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
                /* 111 */ &[Matc, GapX, GapY][..],
            ],
            rng,
        }
    }

    pub(crate) fn fill_matrices(
        &mut self,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<&dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) {
        self.init_x(left_info, left_scoring);
        self.init_y(right_info, right_scoring);
        for i in 1..self.rows {
            for j in 1..self.cols {
                if left_info[i - 1].flag == GapFixed || right_info[j - 1].flag == GapFixed {
                    (
                        self.score.m[i][j],
                        self.score.x[i][j],
                        self.score.y[i][j],
                        self.trace.m[i][j],
                    ) = self.fixed_gap_either(
                        i,
                        j,
                        left_info[i - 1].flag == GapFixed,
                        right_info[j - 1].flag == GapFixed,
                    );
                    self.trace.x[i][j] = self.trace.m[i][j];
                    self.trace.y[i][j] = self.trace.m[i][j];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) = self.possible_match(
                        i - 1,
                        j - 1,
                        left_info,
                        left_scoring,
                        right_info,
                        right_scoring,
                    );
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.possible_gap_x(i - 1, j, left_info, left_scoring, right_info);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, right_info, right_scoring, left_info);
                }
            }
        }
    }

    pub(crate) fn traceback(
        &self,
        left_child_info: &[ParsimonySiteInfo],
        right_child_info: &[ParsimonySiteInfo],
    ) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
        let mut i = self.rows - 1;
        let mut j = self.cols - 1;
        let (pars_score, mut trace) =
            self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]);
        let max_alignment_length = left_child_info.len() + right_child_info.len();
        let mut node_info = Vec::<ParsimonySiteInfo>::with_capacity(max_alignment_length);
        let mut alignment = Alignment::new(
            Mapping::with_capacity(max_alignment_length),
            Mapping::with_capacity(max_alignment_length),
        );
        while i > 0 || j > 0 {
            match trace {
                Matc => {
                    if left_child_info[i - 1].flag == GapFixed
                        && right_child_info[j - 1].flag == GapFixed
                    {
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(None);
                        alignment.map_x.push(None);
                        alignment.map_y.push(Some(j - 1));
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::gap_set(), GapFixed));
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::gap_set(), GapFixed));
                    } else {
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(Some(j - 1));
                        let mut set = &left_child_info[i - 1].set & &right_child_info[j - 1].set;
                        if set.is_empty() {
                            set = &left_child_info[i - 1].set | &right_child_info[j - 1].set;
                        }
                        node_info.push(ParsimonySiteInfo::new(set, NoGap));
                    }
                    trace = self.trace.m[i][j];
                    i -= 1;
                    j -= 1;
                }
                GapX => {
                    alignment.map_x.push(Some(i - 1));
                    alignment.map_y.push(None);
                    match left_child_info[j - 1].flag {
                        GapFixed | GapOpen | GapExt => node_info
                            .push(ParsimonySiteInfo::new(parsimony_sets::gap_set(), GapFixed)),
                        NoGap => node_info.push(ParsimonySiteInfo::new(
                            left_child_info[i - 1].set.clone(),
                            GapOpen,
                        )),
                    }
                    trace = self.trace.x[i][j];
                    i -= 1;
                }
                GapY => {
                    alignment.map_x.push(None);
                    alignment.map_y.push(Some(j - 1));
                    match right_child_info[j - 1].flag {
                        GapFixed | GapOpen | GapExt => node_info
                            .push(ParsimonySiteInfo::new(parsimony_sets::gap_set(), GapFixed)),
                        NoGap => node_info.push(ParsimonySiteInfo::new(
                            right_child_info[j - 1].set.clone(),
                            GapOpen,
                        )),
                    }
                    trace = self.trace.y[i][j];
                    j -= 1;
                }
                Skip => {
                    unreachable!()
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
            match node_info[i - 1].flag {
                GapFixed | GapOpen | GapExt => {}
                NoGap => {
                    if self.score.x[i - 1][0] == 0.0 {
                        self.score.x[i][0] += scoring.gap_open_cost();
                    } else {
                        self.score.x[i][0] += scoring.gap_ext_cost();
                    }
                }
            }
            self.score.y[i][0] = INF;
            self.score.m[i][0] = INF;
            self.trace.m[i][0] = GapX;
            self.trace.x[i][0] = GapX;
            self.trace.y[i][0] = GapX;
        }
    }

    fn init_y(
        &mut self,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<&dyn BranchParsimonyCosts>,
    ) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1];

            match node_info[j - 1].flag {
                GapFixed | GapOpen | GapExt => {}
                NoGap => {
                    if self.score.y[0][j - 1] == 0.0 {
                        self.score.y[0][j] += scoring.gap_open_cost();
                    } else {
                        self.score.y[0][j] += scoring.gap_ext_cost();
                    }
                }
            }
            self.score.x[0][j] = INF;
            self.score.m[0][j] = INF;
            self.trace.m[0][j] = GapY;
            self.trace.x[0][j] = GapY;
            self.trace.y[0][j] = GapY;
        }
    }

    fn possible_gap_x(
        &self,
        i: usize,
        j: usize,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<&dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
    ) -> (f64, Direction) {
        match left_info[i].flag {
            GapOpen | GapFixed => {
                self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j])
            }
            GapExt => {
                let m_gap_adjustment = left_scoring.gap_open_cost() - left_scoring.gap_ext_cost();
                let y_gap_adjustment =
                    self.left_gap_cost_adjustment(i, j, left_info, left_scoring, right_info);
                self.select_direction(
                    self.score.m[i][j] + m_gap_adjustment,
                    self.score.x[i][j],
                    self.score.y[i][j] + y_gap_adjustment,
                )
            }
            NoGap => self.select_direction(
                self.score.m[i][j] + left_scoring.gap_open_cost(),
                self.gap_x_score(i, j, left_info, left_scoring),
                self.score.y[i][j] + left_scoring.gap_open_cost(),
            ),
        }
    }

    fn left_gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<&dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
    ) -> f64 {
        let mut y_gap_adjustment = 0.0;
        let mut skip_i = i;
        let mut skip_j = j;
        while (skip_j > 0 && right_info[skip_j - 1].flag == GapFixed)
            || (skip_i > 0 && left_info[skip_i - 1].flag == GapFixed)
            || self.trace.y[skip_i][skip_j] == GapY
        {
            if right_info[skip_j - 1].flag == GapFixed || self.trace.y[skip_i][skip_j] == GapY {
                skip_j -= 1;
            }
            if left_info[skip_i - 1].flag == GapFixed {
                skip_i -= 1;
            }
        }
        if self.trace.y[skip_i][skip_j] != GapX {
            y_gap_adjustment = left_scoring.gap_open_cost() - left_scoring.gap_ext_cost();
        }
        y_gap_adjustment
    }

    fn possible_gap_y(
        &self,
        i: usize,
        j: usize,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<&dyn BranchParsimonyCosts>,
        left_info: &[ParsimonySiteInfo],
    ) -> (f64, Direction) {
        match right_info[j].flag {
            GapFixed | GapOpen => {
                self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j])
            }
            GapExt => {
                let x_gap_adjustment =
                    self.right_gap_cost_adjustment(i, j, left_info, right_info, right_scoring);
                let m_gap_adjustment = right_scoring.gap_open_cost() - right_scoring.gap_ext_cost();
                self.select_direction(
                    self.score.m[i][j] + m_gap_adjustment,
                    self.score.x[i][j] + x_gap_adjustment,
                    self.score.y[i][j],
                )
            }
            NoGap => self.select_direction(
                self.score.m[i][j] + right_scoring.gap_open_cost(),
                self.score.x[i][j] + right_scoring.gap_open_cost(),
                self.gap_y_score(i, j, right_info, right_scoring),
            ),
        }
    }

    fn right_gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        left_info: &[ParsimonySiteInfo],
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut x_gap_adjustment = 0.0;
        let mut skip_i = i;
        let mut skip_j = j;
        while (skip_j > 0 && right_info[skip_j - 1].flag == GapFixed)
            || (skip_i > 0 && left_info[skip_i - 1].flag == GapFixed)
            || self.trace.x[skip_i][skip_j] == GapX
        {
            if right_info[skip_j - 1].flag == GapFixed || self.trace.x[skip_i][skip_j] == GapX {
                skip_j -= 1;
            }
            if left_info[skip_i - 1].flag == GapFixed {
                skip_i -= 1;
            }
        }
        if self.trace.x[skip_i][skip_j] != GapY {
            x_gap_adjustment = right_scoring.gap_open_cost() - right_scoring.gap_ext_cost();
        }
        x_gap_adjustment
    }

    fn possible_match(
        &self,
        i: usize,
        j: usize,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<&dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        let (x_gap_adjustment, y_gap_adjustment) =
            self.gap_cost_adjustment(i, j, left_info, left_scoring, right_info, right_scoring);
        let score = self.get_match_cost(
            &left_info[i].set,
            left_scoring,
            &right_info[j].set,
            right_scoring,
        );
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
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<&dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> (f64, f64) {
        let mut x_gap_adjustment = 0.0;
        let mut y_gap_adjustment = 0.0;
        if left_info[i].flag != GapExt && right_info[j].flag != GapExt {
        } else {
            if left_info[i].flag == GapExt {
                let skip_j = (1..=j)
                    .rev()
                    .find(|&skip_j| {
                        self.trace.y[i][skip_j] != GapY && right_info[skip_j - 1].flag != GapFixed
                    })
                    .unwrap_or(0);
                if self.trace.y[i][skip_j] != Matc {
                    y_gap_adjustment += left_scoring.gap_open_cost() - left_scoring.gap_ext_cost();
                }
            } else {
                y_gap_adjustment += left_scoring.gap_open_cost() - left_scoring.gap_ext_cost();
            }
            if right_info[j].flag == GapExt {
                let skip_i = (1..=i)
                    .rev()
                    .find(|&skip_i| {
                        self.trace.x[skip_i][j] != GapX && left_info[skip_i - 1].flag != GapFixed
                    })
                    .unwrap_or(0);
                if self.trace.x[skip_i][j] != Matc {
                    x_gap_adjustment +=
                        right_scoring.gap_open_cost() - right_scoring.gap_ext_cost();
                }
            } else {
                x_gap_adjustment += right_scoring.gap_open_cost() - right_scoring.gap_ext_cost();
            }
        }
        (x_gap_adjustment, y_gap_adjustment)
    }

    fn get_match_cost(
        &self,
        lset: &ParsimonySet,
        lscoring: &Box<&dyn BranchParsimonyCosts>,
        rset: &ParsimonySet,
        rscoring: &Box<&dyn BranchParsimonyCosts>,
    ) -> f64 {
        (lset | rset)
            .into_iter()
            .map(|a| min_score(lset, lscoring, a) + min_score(rset, rscoring, a))
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

    fn fixed_gap_either(
        &self,
        i: usize,
        j: usize,
        gap_x: bool,
        gap_y: bool,
    ) -> (f64, f64, f64, Direction) {
        (
            self.score.m[i - gap_x as usize][j - gap_y as usize],
            self.score.x[i - gap_x as usize][j - gap_y as usize],
            self.score.y[i - gap_x as usize][j - gap_y as usize],
            if gap_x && gap_y {
                Matc
            } else if gap_x {
                GapY
            } else {
                GapX
            },
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
            && (node_info[skip_i - 1].flag == GapOpen || node_info[skip_i - 1].flag == GapExt)
            && self.trace.x[skip_i][j] == GapX
        {
            skip_i -= 1;
        }
        if self.trace.x[skip_i][j] != GapX
            && (skip_i == 0
                || node_info[skip_i - 1].flag == GapOpen
                || node_info[skip_i - 1].flag == GapExt)
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
        while skip_j > 0 && node_info[skip_j - 1].flag == GapOpen && self.trace.y[i][skip_j] == GapY
        {
            skip_j -= 1;
        }
        if self.trace.y[i][skip_j] != GapY && (skip_j == 0 || node_info[skip_j - 1].flag == GapOpen)
        {
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
