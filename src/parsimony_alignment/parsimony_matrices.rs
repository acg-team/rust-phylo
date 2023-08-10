use super::parsimony_costs::BranchParsimonyCosts as BranchCosts;
use super::parsimony_info::SiteFlag::{self, GapExt, GapFixed, GapOpen, NoGap};
use super::parsimony_sets::{gap_set, ParsimonySet};
use super::{
    parsimony_info::ParsimonySiteInfo as SiteInfo,
    Direction::{self, GapInX, GapInY, Matc},
};
use crate::alignment::{Alignment, Mapping};
use crate::cmp_f64;
use std::f64::INFINITY as INF;
use std::{fmt, iter::zip};

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
            x: vec![vec![GapInY; len2]; len1],
            y: vec![vec![GapInX; len2]; len1],
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

fn score_match_one_branch(
    a_set: &ParsimonySet,
    c_set: &ParsimonySet,
    c_scor: &Box<&dyn BranchCosts>,
) -> f64 {
    a_set
        .into_iter()
        .map(|&ancestor| min_score(c_set, c_scor, ancestor))
        .min_by(cmp_f64())
        .unwrap()
}

fn score_match_both_branches(
    a_set: &ParsimonySet,
    x_set: &ParsimonySet,
    x_scor: &Box<&dyn BranchCosts>,
    y_set: &ParsimonySet,
    y_scor: &Box<&dyn BranchCosts>,
) -> f64 {
    a_set
        .into_iter()
        .map(|&ancestor| min_score(x_set, x_scor, ancestor) + min_score(y_set, y_scor, ancestor))
        .min_by(cmp_f64())
        .unwrap()
}

fn min_score(set: &ParsimonySet, scor: &Box<&dyn BranchCosts>, ancestor: u8) -> f64 {
    set.into_iter()
        .map(|&child| scor.match_cost(ancestor, child))
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
                /* 010 */ &[GapInY][..],
                /* 011 */ &[Matc, GapInY][..],
                /* 100 */ &[GapInX][..],
                /* 101 */ &[Matc, GapInX][..],
                /* 110 */ &[GapInY, GapInX][..],
                /* 111 */ &[Matc, GapInX, GapInY][..],
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
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) {
        self.init_x(x_info, x_scor, y_scor);
        self.init_y(y_info, x_scor, y_scor);
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
                        self.fill_s_m(i - 1, j - 1, x_info, x_scor, y_info, y_scor);
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.fill_s_x(i - 1, j, x_info, x_scor, y_info, y_scor);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.fill_s_y(i, j - 1, x_info, x_scor, y_info, y_scor);
                }
            }
        }
    }

    fn init_x(
        &mut self,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_scor: &Box<&dyn BranchCosts>,
    ) {
        for i in 1..self.rows {
            self.score.x[i][0] = self.score.x[i - 1][0]
                + if x_info[i - 1].no_gap() {
                    score_match_one_branch(&x_info[i - 1].set, &x_info[i - 1].set, x_scor)
                        + if self.score.x[i - 1][0] == 0.0 {
                            y_scor.gap_open_cost()
                        } else {
                            y_scor.gap_ext_cost()
                        }
                } else {
                    0.0
                };
            if !x_info[i - 1].is_fixed() {
                self.trace.m[i][0] = GapInY;
                self.trace.x[i][0] = GapInY;
                self.trace.y[i][0] = GapInY;
            }
            self.score.y[i][0] = INF;
            self.score.m[i][0] = INF;
        }
    }

    fn init_y(
        &mut self,
        y_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_scor: &Box<&dyn BranchCosts>,
    ) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1]
                + if y_info[j - 1].no_gap() {
                    score_match_one_branch(&y_info[j - 1].set, &y_info[j - 1].set, y_scor)
                        + if self.score.y[0][j - 1] == 0.0 {
                            x_scor.gap_open_cost()
                        } else {
                            x_scor.gap_ext_cost()
                        }
                } else {
                    0.0
                };
            if !y_info[j - 1].is_fixed() {
                self.trace.m[0][j] = GapInX;
                self.trace.x[0][j] = GapInX;
                self.trace.y[0][j] = GapInX;
            }
            self.score.x[0][j] = INF;
            self.score.m[0][j] = INF;
        }
    }

    fn fill_s_m(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> (f64, Direction) {
        let anc_set = if !(&x_info[i].set & &y_info[j].set).is_empty() {
            &x_info[i].set & &y_info[j].set
        } else {
            &x_info[i].set | &y_info[j].set
        };
        let match_score =
            score_match_both_branches(&anc_set, &x_info[i].set, x_scor, &y_info[j].set, y_scor);
        let (x_gap_adj, y_gap_adj) =
            self.score_match_gap_cost_adjustment(i, j, x_info, x_scor, y_info, y_scor);
        self.select_direction(
            self.score.m[i][j] + match_score,
            self.score.x[i][j] + x_gap_adj + match_score,
            self.score.y[i][j] + y_gap_adj + match_score,
        )
    }

    fn score_match_gap_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> (f64, f64) {
        if !x_info[i].is_ext() && !y_info[j].is_ext() {
            return (0.0, 0.0);
        }
        let x_gap_adj = if y_info[j].is_ext() {
            zip(self.trace.x.iter().take(i + 1), x_info.iter().take(i + 1))
                .rev()
                .find(|(dir, info)| dir[j] != GapInY && !info.is_fixed())
                .filter(|(dir, _)| dir[j] != Matc)
                .map_or(0.0, |_| y_scor.gap_open_cost() - y_scor.gap_ext_cost())
        } else {
            y_scor.gap_open_cost() - y_scor.gap_ext_cost()
        };
        let y_gap_adj = if x_info[i].is_ext() {
            zip(
                self.trace.y[i].iter().take(j + 1),
                y_info.iter().take(j + 1),
            )
            .rev()
            .find(|(&dir, info)| dir != GapInX && !info.is_fixed())
            .filter(|(&dir, _)| dir != Matc)
            .map_or(0.0, |_| x_scor.gap_open_cost() - x_scor.gap_ext_cost())
        } else {
            x_scor.gap_open_cost() - x_scor.gap_ext_cost()
        };
        (x_gap_adj, y_gap_adj)
    }

    fn fill_s_x(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> (f64, Direction) {
        let (sm, sx, sy) = match x_info[i].flag {
            GapOpen | GapFixed => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + x_scor.gap_open_cost() - x_scor.gap_ext_cost(),
                self.score.x[i][j],
                self.score.y[i][j] + self.gap_y_cost_adjustment(i, j, x_info, x_scor, y_info),
            ),
            NoGap => {
                let match_score = score_match_one_branch(&x_info[i].set, &x_info[i].set, x_scor);
                (
                    self.score.m[i][j] + match_score + y_scor.gap_open_cost(),
                    self.score.x[i][j] + match_score + self.new_gap_y_score(i, j, x_info, y_scor),
                    self.score.y[i][j] + match_score + y_scor.gap_open_cost(),
                )
            }
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_y_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
    ) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while nj > 0
            && ni > 0
            && (x_info[ni - 1].is_fixed()
                || y_info[nj - 1].is_fixed()
                || self.trace.y[ni][nj] == GapInX)
        {
            ni -= (x_info[ni - 1].is_fixed()) as usize;
            nj -= (y_info[nj - 1].is_fixed() || self.trace.y[ni][nj] == GapInX) as usize;
        }
        if self.trace.y[ni][nj] != GapInY {
            x_scor.gap_open_cost() - x_scor.gap_ext_cost()
        } else {
            0.0
        }
    }

    fn new_gap_y_score(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> f64 {
        zip(
            self.trace.x.iter().take(i + 1).skip(1),
            x_info.iter().take(i + 1),
        )
        .rev()
        .find(|(dir, info)| !(dir[j] == GapInY && info.is_possible()))
        .filter(|(dir, info)| !(dir[j] != GapInY && info.is_possible()))
        .map_or(y_scor.gap_open_cost(), |_| y_scor.gap_ext_cost())
    }

    fn fill_s_y(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> (f64, Direction) {
        let (sm, sx, sy) = match y_info[j].flag {
            GapFixed | GapOpen => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + y_scor.gap_open_cost() - y_scor.gap_ext_cost(),
                self.score.x[i][j] + self.gap_x_cost_adjustment(i, j, x_info, y_info, y_scor),
                self.score.y[i][j],
            ),
            NoGap => {
                let match_score = score_match_one_branch(&y_info[j].set, &y_info[j].set, y_scor);
                (
                    self.score.m[i][j] + match_score + x_scor.gap_open_cost(),
                    self.score.x[i][j] + match_score + x_scor.gap_open_cost(),
                    self.score.y[i][j] + match_score + self.new_x_gap_score(i, j, y_info, x_scor),
                )
            }
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_x_cost_adjustment(
        &self,
        i: usize,
        j: usize,
        x_info: &[SiteInfo],
        y_info: &[SiteInfo],
        y_scor: &Box<&dyn BranchCosts>,
    ) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while ni > 0
            && nj > 0
            && (y_info[nj - 1].is_fixed()
                || x_info[ni - 1].is_fixed()
                || self.trace.x[ni][nj] == GapInY)
        {
            ni -= (x_info[ni - 1].is_fixed() || self.trace.x[ni][nj] == GapInY) as usize;
            nj -= (y_info[nj - 1].is_fixed()) as usize;
        }
        if self.trace.x[ni][nj] != GapInX {
            y_scor.gap_open_cost() - y_scor.gap_ext_cost()
        } else {
            0.0
        }
    }

    fn new_x_gap_score(
        &self,
        i: usize,
        j: usize,
        y_info: &[SiteInfo],
        x_scor: &Box<&dyn BranchCosts>,
    ) -> f64 {
        zip(
            self.trace.y[i].iter().take(j + 1).skip(1),
            y_info.iter().take(j + 1),
        )
        .rev()
        .find(|(&dir, info)| !(dir == GapInX && info.is_possible()))
        .filter(|(&dir, info)| !(dir != GapInY && info.is_possible()))
        .map_or(x_scor.gap_open_cost(), |_| x_scor.gap_ext_cost())
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
                    GapInY => {
                        action = self.trace.x[i][j];
                        i -= 1;
                        let (set, flag) = match x_info[i].flag {
                            GapOpen | GapExt => (gap_set(), GapFixed),
                            NoGap => (x_info[i].set.clone(), self.gap_x_open_or_ext(i, j, x_info)),
                            GapFixed => unreachable!(),
                        };
                        (Some(i), None, set, flag)
                    }
                    GapInX => {
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
        if self.trace.x[i + 1][j] != GapInY
            || i == 0
            || (x_info[i - 1].is_possible() && self.trace.x[i][j] != GapInX)
        {
            GapOpen
        } else {
            GapExt
        }
    }

    fn gap_y_open_or_ext(&self, i: usize, j: usize, y_info: &[SiteInfo]) -> SiteFlag {
        if self.trace.y[i][j + 1] != GapInX
            || j == 0
            || (y_info[j - 1].is_possible() && self.trace.y[i][j] != GapInY)
        {
            GapOpen
        } else {
            GapExt
        }
    }
}

#[cfg(test)]
mod parsimony_matrices_tests;
