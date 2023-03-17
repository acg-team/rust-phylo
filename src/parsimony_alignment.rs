use bio::io::fasta::Record;
use rand::prelude::*;
use std::fmt;

use crate::{
    sequences::{self, SequenceType},
    tree,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapX,
    GapY,
    Skip,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParsAlignSiteInfo {
    set: u8,
    poss_gap: bool,
    perm_gap: bool,
}

impl ParsAlignSiteInfo {
    pub(crate) fn new(set: u8, poss_gap: bool, perm_gap: bool) -> ParsAlignSiteInfo {
        ParsAlignSiteInfo {
            set,
            poss_gap,
            perm_gap,
        }
    }
    pub(crate) fn new_leaf(set: u8) -> ParsAlignSiteInfo {
        ParsAlignSiteInfo::new(set, false, false)
    }
}

pub(crate) struct ScoreMatrices {
    pub(crate) m: Vec<Vec<f32>>,
    pub(crate) x: Vec<Vec<f32>>,
    pub(crate) y: Vec<Vec<f32>>,
}

impl ScoreMatrices {
    pub(crate) fn new(len1: usize, len2: usize) -> ScoreMatrices {
        ScoreMatrices {
            m: vec![vec![0.0; len2]; len1],
            x: vec![vec![0.0; len2]; len1],
            y: vec![vec![0.0; len2]; len1],
        }
    }
}

pub(crate) struct TracebackMatrices {
    pub(crate) m: Vec<Vec<Direction>>,
    pub(crate) x: Vec<Vec<Direction>>,
    pub(crate) y: Vec<Vec<Direction>>,
}

impl TracebackMatrices {
    pub(crate) fn new(len1: usize, len2: usize) -> TracebackMatrices {
        TracebackMatrices {
            m: vec![vec![Direction::Skip; len2]; len1],
            x: vec![vec![Direction::Skip; len2]; len1],
            y: vec![vec![Direction::Skip; len2]; len1],
        }
    }
}

pub(crate) struct ParsimonyAlignmentMatrices {
    rows: usize,
    cols: usize,
    pub(crate) score: ScoreMatrices,
    pub(crate) trace: TracebackMatrices,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    match_cost: f32,
    direction_picker: [&'static [Direction]; 8],
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
        gap_open_cost: f32,
        gap_ext_cost: f32,
        match_cost: f32,
    ) -> ParsimonyAlignmentMatrices {
        ParsimonyAlignmentMatrices {
            rows,
            cols,
            score: ScoreMatrices::new(rows, cols),
            trace: TracebackMatrices::new(rows, cols),
            gap_open_cost,
            gap_ext_cost,
            match_cost,
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
        }
    }

    fn init_x(&mut self, node_info: &[ParsAlignSiteInfo]) {
        for i in 1..self.rows {
            if node_info[i - 1].perm_gap || node_info[i - 1].poss_gap {
                self.score.x[i][0] = self.score.x[i - 1][0];
            } else if self.score.x[i - 1][0] == 0.0 {
                self.score.x[i][0] = self.score.x[i - 1][0] + self.gap_open_cost;
            } else {
                self.score.x[i][0] = self.score.x[i - 1][0] + self.gap_ext_cost;
            }
            self.score.y[i][0] = f32::INFINITY;
            self.score.m[i][0] = f32::INFINITY;
            self.trace.m[i][0] = Direction::GapX;
            self.trace.x[i][0] = Direction::GapX;
            self.trace.y[i][0] = Direction::GapX;
        }
    }

    fn init_y(&mut self, node_info: &[ParsAlignSiteInfo]) {
        for j in 1..self.cols {
            if node_info[j - 1].perm_gap || node_info[j - 1].poss_gap {
                self.score.y[0][j] = self.score.y[0][j - 1];
            } else if self.score.y[0][j - 1] == 0.0 {
                self.score.y[0][j] = self.score.y[0][j - 1] + self.gap_open_cost;
            } else {
                self.score.y[0][j] = self.score.y[0][j - 1] + self.gap_ext_cost;
            }
            self.score.x[0][j] = f32::INFINITY;
            self.score.m[0][j] = f32::INFINITY;
            self.trace.m[0][j] = Direction::GapY;
            self.trace.x[0][j] = Direction::GapY;
            self.trace.y[0][j] = Direction::GapY;
        }
    }

    fn possible_match(
        &self,
        ni: usize,
        nj: usize,
        left_child_info: &[ParsAlignSiteInfo],
        right_child_info: &[ParsAlignSiteInfo],
    ) -> (f32, Direction) {
        let matched = left_child_info[ni].set & right_child_info[nj].set == 0;
        self.select_direction(
            self.score.m[ni][nj] + if matched { self.match_cost } else { 0.0 },
            self.score.x[ni][nj] + if matched { self.match_cost } else { 0.0 },
            self.score.y[ni][nj] + if matched { self.match_cost } else { 0.0 },
        )
    }

    fn possible_gap_y(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsAlignSiteInfo],
    ) -> (f32, Direction) {
        if node_info[nj].perm_gap {
            self.select_direction(
                self.score.m[ni][nj],
                self.score.x[ni][nj],
                self.score.y[ni][nj],
            )
        } else {
            self.select_direction(
                self.score.m[ni][nj] + self.gap_open_cost,
                self.score.x[ni][nj] + self.gap_open_cost,
                self.gap_y_score(ni, nj, node_info),
            )
        }
    }

    fn possible_gap_x(
        &self,
        ni: usize,
        nj: usize,
        node_info: &[ParsAlignSiteInfo],
    ) -> (f32, Direction) {
        if node_info[ni].perm_gap {
            self.select_direction(
                self.score.m[ni][nj],
                self.score.x[ni][nj],
                self.score.y[ni][nj],
            )
        } else {
            self.select_direction(
                self.score.m[ni][nj] + self.gap_open_cost,
                self.gap_x_score(ni, nj, node_info),
                self.score.y[ni][nj] + self.gap_open_cost,
            )
        }
    }

    fn select_direction(&self, sm: f32, sx: f32, sy: f32) -> (f32, Direction) {
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
            self.direction_picker[sel_mat]
                [random::<usize>() % self.direction_picker[sel_mat].len()],
        )
    }

    fn insertion_in_either(
        &self,
        i: usize,
        j: usize,
        insx: &bool,
        insy: &bool,
    ) -> (f32, f32, f32, Direction) {
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

    fn fill_matrices(
        &mut self,
        left_child_info: &[ParsAlignSiteInfo],
        right_child_info: &[ParsAlignSiteInfo],
    ) {
        self.init_x(left_child_info);
        self.init_y(right_child_info);
        for i in 1..self.rows {
            for j in 1..self.cols {
                if left_child_info[i - 1].poss_gap || right_child_info[j - 1].poss_gap {
                    (
                        self.score.m[i][j],
                        self.score.x[i][j],
                        self.score.y[i][j],
                        self.trace.m[i][j],
                    ) = self.insertion_in_either(
                        i,
                        j,
                        &left_child_info[i - 1].poss_gap,
                        &right_child_info[j - 1].poss_gap,
                    );
                    self.trace.x[i][j] = self.trace.m[i][j];
                    self.trace.y[i][j] = self.trace.m[i][j];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) =
                        self.possible_match(i - 1, j - 1, left_child_info, right_child_info);
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.possible_gap_x(i - 1, j, left_child_info);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, right_child_info);
                }
            }
        }
    }

    fn traceback(
        &self,
        left_child_info: &[ParsAlignSiteInfo],
        right_child_info: &[ParsAlignSiteInfo],
    ) -> (Vec<ParsAlignSiteInfo>, Vec<(isize, isize)>) {
        println!("{}", self);

        let mut i = self.rows - 1;
        let mut j = self.cols - 1;
        let (_min_val, start_traceback) =
            self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]);
        let mut trace = match start_traceback {
            Direction::Matc => self.trace.m[i][j],
            Direction::GapX => self.trace.x[i][j],
            Direction::GapY => self.trace.y[i][j],
            Direction::Skip => self.trace.m[i][j],
        };
        println!("{:?}", start_traceback);

        let max_alignment_length = left_child_info.len() + right_child_info.len();
        let mut node_info = Vec::<ParsAlignSiteInfo>::with_capacity(max_alignment_length);
        let mut alignment = Vec::<(isize, isize)>::with_capacity(max_alignment_length);
        while i > 0 || j > 0 {
            println!("{}, {}", i, j);
            match trace {
                Direction::Matc => {
                    println!("Match");
                    if left_child_info[i - 1].perm_gap && right_child_info[j - 1].perm_gap {
                        alignment.push((i as isize - 1, -1));
                        alignment.push((-1, j as isize - 1));

                        node_info.push(ParsAlignSiteInfo {
                            set: 0b10000,
                            poss_gap: true,
                            perm_gap: true,
                        });
                        node_info.push(ParsAlignSiteInfo {
                            set: 0b10000,
                            poss_gap: true,
                            perm_gap: true,
                        });
                    } else {
                        alignment.push((i as isize - 1, j as isize - 1));
                        let mut set = left_child_info[i - 1].set & right_child_info[j - 1].set;
                        if set == 0 {
                            set = left_child_info[i - 1].set | right_child_info[j - 1].set;
                        }
                        node_info.push(ParsAlignSiteInfo {
                            set: set,
                            poss_gap: false,
                            perm_gap: false,
                        });
                    }
                    i -= 1;
                    j -= 1;
                    trace = self.trace.m[i][j];
                }
                Direction::GapX => {
                    println!("GapX");
                    alignment.push((i as isize - 1, -1));
                    if left_child_info[i - 1].perm_gap || left_child_info[i - 1].poss_gap {
                        node_info.push(ParsAlignSiteInfo::new(0b10000, true, true));
                    } else {
                        node_info.push(ParsAlignSiteInfo::new(
                            left_child_info[i - 1].set,
                            true,
                            false,
                        ));
                    }
                    i -= 1;
                    trace = self.trace.x[i][j];
                }
                Direction::GapY => {
                    println!("GapY");
                    alignment.push((-1, j as isize - 1));
                    if right_child_info[j - 1].perm_gap || right_child_info[j - 1].poss_gap {
                        node_info.push(ParsAlignSiteInfo::new(0b10000, true, true));
                    } else {
                        node_info.push(ParsAlignSiteInfo::new(
                            right_child_info[j - 1].set,
                            true,
                            false,
                        ));
                    }
                    j -= 1;
                    trace = self.trace.y[i][j];
                }
                Direction::Skip => {
                    if i != 0 && j != 0 {
                        panic!("We shouldn't be here");
                    }
                }
            }
        }
        (node_info, alignment)
    }

    fn gap_x_score(&self, ni: usize, nj: usize, node_info: &[ParsAlignSiteInfo]) -> f32 {
        let mut skip_index = ni;
        while skip_index > 0
            && node_info[skip_index - 1].perm_gap
            && self.trace.x[skip_index][nj] == Direction::GapX
        {
            skip_index -= 1;
        }
        self.score.x[skip_index][nj] + if self.trace.x[skip_index][nj] != Direction::GapX
            && (skip_index == 0 || node_info[skip_index - 1].perm_gap)
        {
            self.gap_open_cost
        } else {
            self.gap_ext_cost
        }
    }

    fn gap_y_score(&self, ni: usize, nj: usize, node_info: &[ParsAlignSiteInfo]) -> f32 {
        let mut skip_index = nj;
        while skip_index > 0
            && node_info[skip_index - 1].perm_gap
            && self.trace.y[ni][skip_index] == Direction::GapY
        {
            skip_index -= 1;
        }
        if self.trace.y[ni][skip_index] != Direction::GapY
            && (skip_index == 0 || node_info[skip_index - 1].perm_gap)
        {
            self.score.y[ni][skip_index] + self.gap_open_cost
        } else {
            self.score.y[ni][skip_index] + self.gap_ext_cost
        }
    }
}

pub(crate) fn pars_align(
    left_child_info: &[ParsAlignSiteInfo],
    right_child_info: &[ParsAlignSiteInfo],
) -> (Vec<ParsAlignSiteInfo>, Vec<(isize, isize)>) {
    let a = 2.5;
    let b = 0.5;
    let c = 1.0;

    let mut pars_mats = ParsimonyAlignmentMatrices::new(
        left_child_info.len() + 1,
        right_child_info.len() + 1,
        a,
        b,
        c,
    );
    pars_mats.fill_matrices(left_child_info, right_child_info);
    pars_mats.traceback(left_child_info, right_child_info)
}

pub(crate) fn pars_align_on_tree(
    tree: &tree::Tree,
    sequences: &[Record],
    sequence_type: &SequenceType,
) {
    let num = tree.nodes.len();
    let order = &tree.postorder;

    assert_eq!(num, order.len());

    let mut node_info = vec![Vec::<ParsAlignSiteInfo>::new(); num];
    let mut alignments = vec![Vec::<(isize, isize)>::new(); num];

    for &node_idx in order {
        println!("{}", node_idx);
        if tree.is_leaf(node_idx) {
            let pars_sets = sequences::get_parsimony_sets(&sequences[node_idx], sequence_type);
            node_info[node_idx] = pars_sets
                .into_iter()
                .map(|set| ParsAlignSiteInfo::new_leaf(set))
                .collect();
        } else {
            let ch1_idx = tree.nodes[node_idx].children[0];
            let ch2_idx = tree.nodes[node_idx].children[1];

            let (info, alignment) = pars_align(&node_info[ch1_idx], &node_info[ch2_idx]);
            node_info[node_idx] = info;
            alignments[node_idx] = alignment;
        }
    }
}

#[test]
fn pars_align_matrix_simple_test() {
    let a = 2.5;
    let b = 0.5;
    let c = 1.0;

    let node_info_1 = vec![
        ParsAlignSiteInfo::new(0b0100, false, false),
        ParsAlignSiteInfo::new(0b0100, false, false),
    ];
    let node_info_2 = vec![
        ParsAlignSiteInfo::new(0b1000, false, false),
        ParsAlignSiteInfo::new(0b0100, false, false),
    ];

    let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, a, b, c);

    pars_mats.fill_matrices(&node_info_1, &node_info_2);
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, f32::INFINITY, f32::INFINITY],
            vec![f32::INFINITY, 1.0, 2.5],
            vec![f32::INFINITY, 3.5, 1.0]
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, f32::INFINITY, f32::INFINITY],
            vec![2.5, 5.0, 5.5],
            vec![3.0, 3.5, 5.0]
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 2.5, 3.0],
            vec![f32::INFINITY, 5.0, 3.5],
            vec![f32::INFINITY, 5.5, 6.0]
        ]
    );
    let any_dir = vec![Direction::Matc, Direction::GapX, Direction::GapY];
    assert_eq!(
        pars_mats.trace.m[0],
        vec![Direction::Skip, Direction::GapY, Direction::GapY]
    );
    assert_eq!(pars_mats.trace.m[1][0], Direction::GapX);
    assert!(any_dir.contains(&pars_mats.trace.m[1][1]));
    assert_eq!(pars_mats.trace.m[1][2], Direction::GapY);
    assert_eq!(
        pars_mats.trace.m[2],
        vec![Direction::GapX, Direction::GapX, Direction::Matc]
    );

    assert_eq!(
        pars_mats.trace.x[0],
        vec![Direction::Skip, Direction::GapY, Direction::GapY]
    );
    assert_eq!(
        pars_mats.trace.x[1],
        vec![Direction::GapX, Direction::GapY, Direction::GapY]
    );
    assert_eq!(
        pars_mats.trace.x[2],
        vec![Direction::GapX, Direction::Matc, Direction::Matc]
    );

    assert_eq!(
        pars_mats.trace.y[0],
        vec![Direction::Skip, Direction::GapY, Direction::GapY]
    );
    assert_eq!(
        pars_mats.trace.y[1],
        vec![Direction::GapX, Direction::GapX, Direction::Matc]
    );
    assert_eq!(pars_mats.trace.y[2][0], Direction::GapX);
    assert_eq!(pars_mats.trace.y[2][1], Direction::GapX);
    assert!(any_dir.contains(&pars_mats.trace.y[2][2]));
}
