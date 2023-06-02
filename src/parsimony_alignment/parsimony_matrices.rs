use crate::alignment::{Alignment, Mapping};
use crate::parsimony_alignment::parsimony_sets::{self, EMPTY_SET};
use crate::parsimony_alignment::{parsimony_info::ParsimonySiteInfo, Direction};
use std::fmt;

use super::parsimony_costs::BranchParsimonyCosts;

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
            m: vec![vec![Direction::Skip; len2]; len1],
            x: vec![vec![Direction::Skip; len2]; len1],
            y: vec![vec![Direction::Skip; len2]; len1],
        }
    }
}

pub(super) struct ParsimonyAlignmentMatrices {
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
                /* 001 */ &[Direction::Matc][..],
                /* 010 */ &[Direction::GapX][..],
                /* 011 */ &[Direction::Matc, Direction::GapX][..],
                /* 100 */ &[Direction::GapY][..],
                /* 101 */ &[Direction::Matc, Direction::GapY][..],
                /* 110 */ &[Direction::GapX, Direction::GapY][..],
                /* 111 */ &[Direction::Matc, Direction::GapX, Direction::GapY][..],
            ],
            rng,
        }
    }

    pub(crate) fn fill_matrices(
        &mut self,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<dyn BranchParsimonyCosts>,
    ) {
        self.init_x(left_info, left_scoring);
        self.init_y(right_info, right_scoring);
        for i in 1..self.rows {
            for j in 1..self.cols {
                if left_info[i - 1].perm_gap || right_info[j - 1].perm_gap {
                    (
                        self.score.m[i][j],
                        self.score.x[i][j],
                        self.score.y[i][j],
                        self.trace.m[i][j],
                    ) = self.insertion_in_either(
                        i,
                        j,
                        &left_info[i - 1].perm_gap,
                        &right_info[j - 1].perm_gap,
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
                        self.possible_gap_x(i - 1, j, left_info, left_scoring);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, right_info, right_scoring);
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
                Direction::Matc => {
                    if left_child_info[i - 1].perm_gap && right_child_info[j - 1].perm_gap {
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(None);
                        alignment.map_x.push(None);
                        alignment.map_y.push(Some(j - 1));
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::GAP_SET, true, true));
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::GAP_SET, true, true));
                    } else {
                        alignment.map_x.push(Some(i - 1));
                        alignment.map_y.push(Some(j - 1));
                        let mut set = left_child_info[i - 1].set & right_child_info[j - 1].set;
                        if set == EMPTY_SET {
                            set = left_child_info[i - 1].set | right_child_info[j - 1].set;
                        }
                        node_info.push(ParsimonySiteInfo::new(set, false, false));
                    }
                    trace = self.trace.m[i][j];
                    i -= 1;
                    j -= 1;
                }
                Direction::GapX => {
                    alignment.map_x.push(Some(i - 1));
                    alignment.map_y.push(None);
                    if left_child_info[i - 1].perm_gap || left_child_info[i - 1].poss_gap {
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::GAP_SET, true, true));
                    } else {
                        node_info.push(ParsimonySiteInfo::new(
                            left_child_info[i - 1].set,
                            true,
                            false,
                        ));
                    }
                    trace = self.trace.x[i][j];
                    i -= 1;
                }
                Direction::GapY => {
                    alignment.map_x.push(None);
                    alignment.map_y.push(Some(j - 1));
                    if right_child_info[j - 1].perm_gap || right_child_info[j - 1].poss_gap {
                        node_info.push(ParsimonySiteInfo::new(parsimony_sets::GAP_SET, true, true));
                    } else {
                        node_info.push(ParsimonySiteInfo::new(
                            right_child_info[j - 1].set,
                            true,
                            false,
                        ));
                    }
                    trace = self.trace.y[i][j];
                    j -= 1;
                }
                Direction::Skip => {
                    unreachable!()
                }
            }
        }
        node_info.reverse();
        alignment.map_x.reverse();
        alignment.map_y.reverse();
        (node_info, alignment, pars_score)
    }

    fn init_x(&mut self, node_info: &[ParsimonySiteInfo], scoring: &Box<dyn BranchParsimonyCosts>) {
        for i in 1..self.rows {
            self.score.x[i][0] = self.score.x[i - 1][0];
            if node_info[i - 1].perm_gap || node_info[i - 1].poss_gap {
            } else if self.score.x[i - 1][0] == 0.0 {
                self.score.x[i][0] += scoring.gap_open_cost();
            } else {
                self.score.x[i][0] += scoring.gap_ext_cost();
            }
            self.score.y[i][0] = f64::INFINITY;
            self.score.m[i][0] = f64::INFINITY;
            self.trace.m[i][0] = Direction::GapX;
            self.trace.x[i][0] = Direction::GapX;
            self.trace.y[i][0] = Direction::GapX;
        }
    }

    fn init_y(&mut self, node_info: &[ParsimonySiteInfo], scoring: &Box<dyn BranchParsimonyCosts>) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1];
            if node_info[j - 1].perm_gap || node_info[j - 1].poss_gap {
            } else if self.score.y[0][j - 1] == 0.0 {
                self.score.y[0][j] += scoring.gap_open_cost();
            } else {
                self.score.y[0][j] += scoring.gap_ext_cost();
            }
            self.score.x[0][j] = f64::INFINITY;
            self.score.m[0][j] = f64::INFINITY;
            self.trace.m[0][j] = Direction::GapY;
            self.trace.x[0][j] = Direction::GapY;
            self.trace.y[0][j] = Direction::GapY;
        }
    }

    fn possible_gap_x(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        if node_info[ni].poss_gap {
            self.select_direction(
                self.score.m[ni][nj],
                self.score.x[ni][nj],
                self.score.y[ni][nj],
            )
        } else {
            self.select_direction(
                self.score.m[ni][nj] + scoring.gap_open_cost(),
                self.gap_x_score(ni, nj, node_info, scoring),
                self.score.y[ni][nj] + scoring.gap_open_cost(),
            )
        }
    }

    fn possible_gap_y(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        if node_info[nj].poss_gap {
            self.select_direction(
                self.score.m[ni][nj],
                self.score.x[ni][nj],
                self.score.y[ni][nj],
            )
        } else {
            self.select_direction(
                self.score.m[ni][nj] + scoring.gap_open_cost(),
                self.score.x[ni][nj] + scoring.gap_open_cost(),
                self.gap_y_score(ni, nj, node_info, scoring),
            )
        }
    }

    fn possible_match(
        &self,
        ni: usize,
        nj: usize,
        left_info: &[ParsimonySiteInfo],
        left_scoring: &Box<dyn BranchParsimonyCosts>,
        right_info: &[ParsimonySiteInfo],
        right_scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> (f64, Direction) {
        if left_info[ni].set & right_info[nj].set != 0 {
            self.select_direction(
                self.score.m[ni][nj],
                self.score.x[ni][nj],
                self.score.y[ni][nj],
            )
        } else {
            let score = self.get_match_cost(
                left_info[ni].set,
                left_scoring,
                right_info[nj].set,
                right_scoring,
            );
            self.select_direction(
                self.score.m[ni][nj] + score,
                self.score.x[ni][nj] + score,
                self.score.y[ni][nj] + score,
            )
        }
    }

    fn get_match_cost(
        &self,
        left_set: u32,
        left_scoring: &Box<dyn BranchParsimonyCosts>,
        right_set: u32,
        right_scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> f64 {
        left_scoring.match_cost(1, 2)
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

    fn insertion_in_either(
        &self,
        i: usize,
        j: usize,
        insx: &bool,
        insy: &bool,
    ) -> (f64, f64, f64, Direction) {
        let ni = i - *insx as usize;
        let nj = j - *insy as usize;
        (
            self.score.m[ni][nj],
            self.score.x[ni][nj],
            self.score.y[ni][nj],
            if *insx && *insy {
                Direction::Matc
            } else if *insx {
                Direction::GapY
            } else {
                Direction::GapX
            },
        )
    }

    fn gap_x_score(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut skip_index = ni;
        while skip_index > 0
            && node_info[skip_index - 1].poss_gap
            && self.trace.x[skip_index][nj] == Direction::GapX
        {
            skip_index -= 1;
        }
        if self.trace.x[skip_index][nj] != Direction::GapX
            && (skip_index == 0 || node_info[skip_index - 1].poss_gap)
        {
            self.score.x[skip_index][nj] + scoring.gap_open_cost()
        } else {
            self.score.x[skip_index][nj] + scoring.gap_ext_cost()
        }
    }

    fn gap_y_score(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsimonySiteInfo],
        scoring: &Box<dyn BranchParsimonyCosts>,
    ) -> f64 {
        let mut skip_index = nj;
        while skip_index > 0
            && node_info[skip_index - 1].poss_gap
            && self.trace.y[ni][skip_index] == Direction::GapY
        {
            skip_index -= 1;
        }
        if self.trace.y[ni][skip_index] != Direction::GapY
            && (skip_index == 0 || node_info[skip_index - 1].poss_gap)
        {
            self.score.y[ni][skip_index] + scoring.gap_open_cost()
        } else {
            self.score.y[ni][skip_index] + scoring.gap_ext_cost()
        }
    }
}

#[cfg(test)]
mod parsimony_matrices_tests {
    use crate::parsimony_alignment::{
        parsimony_costs::{ParsimonyCosts, ParsimonyCostsSimple},
        parsimony_info::ParsimonySiteInfo,
        parsimony_matrices::ParsimonyAlignmentMatrices,
        Direction,
    };

    #[test]
    fn fill_matrix() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.5;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let node_info_1 = vec![
            ParsimonySiteInfo::new(0b00100, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];
        let node_info_2 = vec![
            ParsimonySiteInfo::new(0b01000, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];

        let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |_| 0);

        pars_mats.fill_matrices(
            &node_info_1,
            &scoring.get_branch_costs(1.0),
            &node_info_2,
            &scoring.get_branch_costs(1.0),
        );

        assert_eq!(
            pars_mats.score.m,
            vec![
                vec![0.0, f64::INFINITY, f64::INFINITY],
                vec![f64::INFINITY, 1.0, 2.5],
                vec![f64::INFINITY, 3.5, 1.0]
            ]
        );
        assert_eq!(
            pars_mats.score.x,
            vec![
                vec![0.0, f64::INFINITY, f64::INFINITY],
                vec![2.5, 5.0, 5.5],
                vec![3.0, 3.5, 5.0]
            ]
        );
        assert_eq!(
            pars_mats.score.y,
            vec![
                vec![0.0, 2.5, 3.0],
                vec![f64::INFINITY, 5.0, 3.5],
                vec![f64::INFINITY, 5.5, 6.0]
            ]
        );
        assert_eq!(
            pars_mats.trace.m,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::Matc, Direction::GapY],
                vec![Direction::GapX, Direction::GapX, Direction::Matc]
            ]
        );
        assert_eq!(
            pars_mats.trace.x,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::Matc, Direction::Matc]
            ]
        );
        assert_eq!(
            pars_mats.trace.y,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapX, Direction::Matc],
                vec![Direction::GapX, Direction::GapX, Direction::Matc]
            ]
        );
    }

    #[test]
    fn fill_matrix_other_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.5;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let node_info_1 = vec![
            ParsimonySiteInfo::new(0b00100, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];
        let node_info_2 = vec![
            ParsimonySiteInfo::new(0b01000, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];

        let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |l| l - 1);
        pars_mats.fill_matrices(
            &node_info_1,
            &scoring.get_branch_costs(1.0),
            &node_info_2,
            &scoring.get_branch_costs(1.0),
        );

        assert_eq!(
            pars_mats.score.m,
            vec![
                vec![0.0, f64::INFINITY, f64::INFINITY],
                vec![f64::INFINITY, 1.0, 2.5],
                vec![f64::INFINITY, 3.5, 1.0]
            ]
        );
        assert_eq!(
            pars_mats.score.x,
            vec![
                vec![0.0, f64::INFINITY, f64::INFINITY],
                vec![2.5, 5.0, 5.5],
                vec![3.0, 3.5, 5.0]
            ]
        );
        assert_eq!(
            pars_mats.score.y,
            vec![
                vec![0.0, 2.5, 3.0],
                vec![f64::INFINITY, 5.0, 3.5],
                vec![f64::INFINITY, 5.5, 6.0]
            ]
        );
        assert_eq!(
            pars_mats.trace.m,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapX, Direction::Matc]
            ]
        );
        assert_eq!(
            pars_mats.trace.x,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::Matc, Direction::Matc]
            ]
        );
        assert_eq!(
            pars_mats.trace.y,
            vec![
                vec![Direction::Skip, Direction::GapY, Direction::GapY],
                vec![Direction::GapX, Direction::GapX, Direction::Matc],
                vec![Direction::GapX, Direction::GapX, Direction::GapY]
            ]
        );
    }

    #[test]
    fn traceback_correct() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.5;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let node_info_1 = vec![
            ParsimonySiteInfo::new(0b00100, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];
        let node_info_2 = vec![
            ParsimonySiteInfo::new(0b01000, false, false),
            ParsimonySiteInfo::new(0b00100, false, false),
        ];
        let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |l| l - 1);
        pars_mats.fill_matrices(
            &node_info_1,
            &scoring.get_branch_costs(1.0),
            &node_info_2,
            &scoring.get_branch_costs(1.0),
        );

        let (node_info, alignment, score) = pars_mats.traceback(&node_info_1, &node_info_2);
        assert_eq!(
            node_info[0],
            ParsimonySiteInfo::new(0b00100 + 0b01000, false, false)
        );
        assert_eq!(node_info[1], ParsimonySiteInfo::new(0b00100, false, false));
        assert_eq!(alignment.map_x, vec![Some(0), Some(1)]);
        assert_eq!(alignment.map_y, vec![Some(0), Some(1)]);
        assert_eq!(score, 1.0);

        let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |_| 0);
        pars_mats.fill_matrices(
            &node_info_1,
            &scoring.get_branch_costs(1.0),
            &node_info_2,
            &scoring.get_branch_costs(1.0),
        );

        let (node_info, alignment, score) = pars_mats.traceback(&node_info_1, &node_info_2);
        assert_eq!(
            node_info[0],
            ParsimonySiteInfo::new(0b00100 + 0b01000, false, false)
        );
        assert_eq!(node_info[1], ParsimonySiteInfo::new(0b00100, false, false));
        assert_eq!(alignment.map_x, vec![Some(0), Some(1)]);
        assert_eq!(alignment.map_y, vec![Some(0), Some(1)]);
        assert_eq!(score, 1.0);
    }
}
