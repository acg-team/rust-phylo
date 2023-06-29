use crate::alignment::{Alignment, Mapping};
use crate::cmp_f64;
use crate::parsimony_alignment::{
    parsimony_info::ParsimonySiteInfo as SiteInfo,
    Direction::{self, GapX, GapY, Matc},
};
use std::f64::INFINITY as INF;
use std::fmt;

use super::parsimony_costs::BranchParsimonyCosts as BranchCosts;
use super::parsimony_info::SiteFlag::{self, GapExt, GapFixed, GapOpen, NoGap};
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

fn min_score(set: &ParsimonySet, scoring: &Box<&dyn BranchCosts>, ancestor: u8) -> f64 {
    set.into_iter()
        .map(|l: &u8| scoring.match_cost(ancestor, *l))
        .min_by(cmp_f64())
        .unwrap()
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

    pub(crate) fn fill_matrices(
        &mut self,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) {
        self.init_x(x_info, x_scoring);
        self.init_y(y_info, y_scoring);
        for i in 1..self.rows {
            for j in 1..self.cols {
                if x_info[i - 1].is_fixed() || y_info[j - 1].is_fixed() {
                    let ni = i - x_info[i - 1].is_fixed() as usize;
                    let nj = j - y_info[j - 1].is_fixed() as usize;
                    self.score.m[i][j] = self.score.m[ni][nj];
                    self.score.x[i][j] = self.score.x[ni][nj];
                    self.score.y[i][j] = self.score.y[ni][nj];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) =
                        self.possible_match(i - 1, j - 1, x_info, x_scoring, y_info, y_scoring);
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.possible_gap_x(i - 1, j, x_info, x_scoring, y_info);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, x_info, y_info, y_scoring);
                }
            }
        }
    }

    fn init_x(&mut self, x_info: &[SiteInfo], x_scoring: &Box<&dyn BranchCosts>) {
        for i in 1..self.rows {
            self.score.x[i][0] = self.score.x[i - 1][0];
            if x_info[i - 1].no_gap() {
                if self.score.x[i - 1][0] == 0.0 {
                    self.score.x[i][0] += x_scoring.gap_open_cost();
                } else {
                    self.score.x[i][0] += x_scoring.gap_ext_cost();
                }
            }
            if !x_info[i - 1].is_fixed() {
                self.trace.m[i][0] = GapX;
                self.trace.x[i][0] = GapX;
                self.trace.y[i][0] = GapX;
            }
            self.score.y[i][0] = INF;
            self.score.m[i][0] = INF;
        }
    }

    fn init_y(&mut self, y_info: &[SiteInfo], y_scoring: &Box<&dyn BranchCosts>) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1];
            if y_info[j - 1].no_gap() {
                if self.score.y[0][j - 1] == 0.0 {
                    self.score.y[0][j] += y_scoring.gap_open_cost();
                } else {
                    self.score.y[0][j] += y_scoring.gap_ext_cost();
                }
            }
            if !y_info[j - 1].is_fixed(){
                self.trace.m[0][j] = GapY;
                self.trace.x[0][j] = GapY;
                self.trace.y[0][j] = GapY;
            }
            self.score.x[0][j] = INF;
            self.score.m[0][j] = INF;
        }
    }

    fn possible_match(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> (f64, Direction) {
        let score = self.get_match_cost(&x_info[i].set, x_scoring, &y_info[j].set, y_scoring);
        let (x_gap_adj, y_gap_adj) =
            self.gap_cost_adjustment(i, j, x_info, x_scoring, y_info, y_scoring);
        self.select_direction(
            self.score.m[i][j] + score,
            self.score.x[i][j] + x_gap_adj + score,
            self.score.y[i][j] + y_gap_adj + score,
        )
    }

    fn get_match_cost(
        &self,
        x_set: &ParsimonySet,
        x_scoring: &Box<&dyn BranchCosts>,
        y_set: &ParsimonySet,
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> f64 {
        (x_set | y_set)
            .into_iter()
            .map(|a| min_score(x_set, x_scoring, a) + min_score(y_set, y_scoring, a))
            .min_by(cmp_f64())
            .unwrap_or(INF)
    }

    fn gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> (f64, f64) {
        if !x_info[i].is_ext() && !y_info[j].is_ext() {
            (0.0, 0.0)
        } else {
            let x_gap_adj = if y_info[j].is_ext() {
                let ni = (1..=i)
                    .rev()
                    .find(|&ni| self.trace.x[ni][j] != GapX && !x_info[ni - 1].is_fixed())
                    .unwrap_or(1);
                if self.trace.x[ni][j] != Matc {
                    y_scoring.gap_open_cost() - y_scoring.gap_ext_cost()
                } else {
                    0.0
                }
            } else {
                y_scoring.gap_open_cost() - y_scoring.gap_ext_cost()
            };
            let y_gap_adj = if x_info[i].is_ext() {
                let nj = (1..=j)
                    .rev()
                    .find(|&nj| self.trace.y[i][nj] != GapY && !y_info[nj - 1].is_fixed())
                    .unwrap_or(1);
                if self.trace.y[i][nj] != Matc {
                    x_scoring.gap_open_cost() - x_scoring.gap_ext_cost()
                } else {
                    0.0
                }
            } else {
                x_scoring.gap_open_cost() - x_scoring.gap_ext_cost()
            };
            (x_gap_adj, y_gap_adj)
        }
    }

    fn possible_gap_x(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
    ) -> (f64, Direction) {
        let (sm, sx, sy) = match x_info[i].flag {
            GapOpen | GapFixed => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + x_scoring.gap_open_cost() - x_scoring.gap_ext_cost(),
                self.score.x[i][j],
                self.score.y[i][j] + self.gap_y_cost_adjustment(i, j, x_info, x_scoring, y_info),
            ),
            NoGap => (
                self.score.m[i][j] + x_scoring.gap_open_cost(),
                self.gap_x_score(i, j, x_info, x_scoring),
                self.score.y[i][j] + x_scoring.gap_open_cost(),
            ),
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_y_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
    ) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while nj > 0
            && ni > 0
            && (x_info[ni - 1].is_fixed()
                || y_info[nj - 1].is_fixed()
                || self.trace.y[ni][nj] == GapY)
        {
            ni -= (x_info[ni - 1].is_fixed()) as usize;
            nj -= (y_info[nj - 1].is_fixed() || self.trace.y[ni][nj] == GapY) as usize;
        }
        if self.trace.y[ni][nj] != GapX {
            x_scoring.gap_open_cost() - x_scoring.gap_ext_cost()
        } else {
            0.0
        }
    }

    fn gap_x_score(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scoring: &Box<&dyn BranchCosts>,
    ) -> f64 {
        let mut ni = i;
        while ni > 0 && (x_info[ni - 1].is_possible()) && self.trace.x[ni][j] == GapX {
            ni -= 1;
        }
        self.score.x[ni][j]
            + if self.trace.x[ni][j] != GapX && (ni == 0 || x_info[ni - 1].is_possible()) {
                x_scoring.gap_open_cost()
            } else {
                x_scoring.gap_ext_cost()
            }
    }

    fn possible_gap_y(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> (f64, Direction) {
        let (sm, sx, sy) = match y_info[j].flag {
            GapFixed | GapOpen => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + y_scoring.gap_open_cost() - y_scoring.gap_ext_cost(),
                self.score.x[i][j] + self.gap_x_cost_adjustment(i, j, x_info, y_info, y_scoring),
                self.score.y[i][j],
            ),
            NoGap => (
                self.score.m[i][j] + y_scoring.gap_open_cost(),
                self.score.x[i][j] + y_scoring.gap_open_cost(),
                self.gap_y_score(i, j, y_info, y_scoring),
            ),
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_x_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while ni > 0
            && nj > 0
            && (y_info[nj - 1].is_fixed()
                || x_info[ni - 1].is_fixed()
                || self.trace.x[ni][nj] == GapX)
        {
            ni -= (x_info[ni - 1].is_fixed() || self.trace.x[ni][nj] == GapX) as usize;
            nj -= (y_info[nj - 1].is_fixed()) as usize;
        }
        if self.trace.x[ni][nj] != GapY {
            y_scoring.gap_open_cost() - y_scoring.gap_ext_cost()
        } else {
            0.0
        }
    }

    fn gap_y_score(
        &self,
        i: usize,
        j: usize,
        y_info: &[SiteInfo],
        y_scoring: &Box<&dyn BranchCosts>,
    ) -> f64 {
        let mut nj = j;
        while nj > 0 && y_info[nj - 1].is_possible() && self.trace.y[i][nj] == GapY {
            nj -= 1;
        }
        self.score.y[i][nj]
            + if self.trace.y[i][nj] != GapY && (nj == 0 || y_info[nj - 1].is_possible()) {
                y_scoring.gap_open_cost()
            } else {
                y_scoring.gap_ext_cost()
            }
    }

    pub(crate) fn traceback(
        &self,
        x_info: &[SiteInfo],
        y_info: &[SiteInfo],
    ) -> (Vec<SiteInfo>, Alignment, f64) {
        let mut i = self.rows - 1;
        let mut j = self.cols - 1;
        let (pars_score, mut action) =
            self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]);
        let max_alignment_length = x_info.len() + y_info.len();
        let mut node_info = Vec::<SiteInfo>::with_capacity(max_alignment_length);
        let mut alignment = Alignment::new(
            Mapping::with_capacity(max_alignment_length),
            Mapping::with_capacity(max_alignment_length),
        );
        while i > 0 || j > 0 {
            if (i > 0 && x_info[i - 1].is_fixed()) || (j > 0 && y_info[j - 1].is_fixed()) {
                if i > 0 && x_info[i - 1].is_fixed() {
                    i -= 1;
                    alignment.map_x.push(Some(i));
                    alignment.map_y.push(None);
                    node_info.push(SiteInfo::new(gap_set(), GapFixed));
                }
                if j > 0 && y_info[j - 1].is_fixed() {
                    j -= 1;
                    alignment.map_x.push(None);
                    alignment.map_y.push(Some(j));
                    node_info.push(SiteInfo::new(gap_set(), GapFixed));
                }
            } else {
                let (map_x, map_y, set, flag) = match action {
                    Matc => {
                        action = self.trace.m[i][j];
                        i -= 1;
                        j -= 1;
                        let mut set = &x_info[i].set & &y_info[j].set;
                        if set.is_empty() {
                            set = &x_info[i].set | &y_info[j].set;
                        }
                        (Some(i), Some(j), set, NoGap)
                    }
                    GapX => {
                        action = self.trace.x[i][j];
                        i -= 1;
                        let (set, flag) = match x_info[i].flag {
                            GapOpen | GapExt => (gap_set(), GapFixed),
                            NoGap => (x_info[i].set.clone(), self.gap_x_open_or_ext(i, j, x_info)),
                            GapFixed => unreachable!(),
                        };
                        (Some(i), None, set, flag)
                    }
                    GapY => {
                        action = self.trace.y[i][j];
                        j -= 1;
                        let (set, flag) = match y_info[j].flag {
                            GapOpen | GapExt => (gap_set(), GapFixed),
                            NoGap => (y_info[j].set.clone(), self.gap_y_open_or_ext(i, j, y_info)),
                            GapFixed => unreachable!(),
                        };
                        (None, Some(j), set, flag)
                    }
                };
                node_info.push(SiteInfo::new(set, flag));
                alignment.map_x.push(map_x);
                alignment.map_y.push(map_y);
            }
        }
        node_info.reverse();
        alignment.map_x.reverse();
        alignment.map_y.reverse();
        (node_info, alignment, pars_score)
    }

    fn gap_x_open_or_ext(&self, i: usize, j: usize, x_info: &[SiteInfo]) -> SiteFlag {
        if self.trace.x[i + 1][j] != GapX
            || i == 0
            || (x_info[i - 1].is_possible() && self.trace.x[i][j] != GapY)
        {
            GapOpen
        } else {
            GapExt
        }
    }

    fn gap_y_open_or_ext(&self, i: usize, j: usize, y_info: &[SiteInfo]) -> SiteFlag {
        if self.trace.y[i][j + 1] != GapY
            || j == 0
            || (y_info[j - 1].is_possible() && self.trace.y[i][j] != GapX)
        {
            GapOpen
        } else {
            GapExt
        }
    }
}

#[cfg(test)]
mod parsimony_matrices_tests;
