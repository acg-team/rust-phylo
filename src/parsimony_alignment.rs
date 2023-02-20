use std::fmt;
use rand::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapX,
    GapY,
    Ins,
    Skip,
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
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) score: ScoreMatrices,
    pub(crate) trace: TracebackMatrices,
    pub(crate) gap_open_cost: f32,
    pub(crate) gap_ext_cost: f32,
    pub(crate) match_cost: f32,
    pub(crate) direction_picker: [&'static [Direction]; 8],
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

    fn init_x(&mut self, ins: &Vec<bool>, del: &Vec<bool>) {
        for i in 1..self.rows {
            if del[i - 1] | ins[i - 1] {
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

    fn init_y(&mut self, ins: &Vec<bool>, del: &Vec<bool>) {
        for j in 1..self.cols {
            if del[j - 1] | ins[j - 1] {
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

    fn possible_match(&self, ni: usize, nj: usize, matched: bool) -> (f32, Direction) {
        self.select_direction(
            self.score.m[ni][nj] + if matched { self.match_cost } else { 0.0 },
            self.score.x[ni][nj] + if matched { self.match_cost } else { 0.0 },
            self.score.y[ni][nj] + if matched { self.match_cost } else { 0.0 },
        )
    }

    fn possible_gap_y(&self, ni: usize, nj: usize, dely: bool) -> (f32, Direction) {
        self.select_direction(
            self.score.m[ni][nj] + if dely { 0.0 } else { self.gap_open_cost },
            self.score.x[ni][nj] + if dely { 0.0 } else { self.gap_open_cost },
            self.score.y[ni][nj] + if dely { 0.0 } else { self.gap_ext_cost },
        )
    }

    fn possible_gap_x(&self, ni: usize, nj: usize, delx: bool) -> (f32, Direction) {
        self.select_direction(
            self.score.m[ni][nj] + if delx { 0.0 } else { self.gap_open_cost },
            self.score.x[ni][nj] + if delx { 0.0 } else { self.gap_ext_cost },
            self.score.y[ni][nj] + if delx { 0.0 } else { self.gap_open_cost },
        )
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
        insx: &Vec<bool>,
        insy: &Vec<bool>,
    ) -> (f32, f32, f32, Direction) {
        let ni = i - if insx[i - 1] { 1 } else { 0 };
        let nj = j - if insy[j - 1] { 1 } else { 0 };
        (
            self.score.m[ni][nj],
            self.score.x[ni][nj],
            self.score.y[ni][nj],
            Direction::Ins,
        )
    }

    fn align_sequences(
        &mut self,
        setsx: &[u8],
        setsy: &[u8],
        insx: &Vec<bool>,
        delx: &Vec<bool>,
        insy: &Vec<bool>,
        dely: &Vec<bool>,
    ) {
        for i in 1..self.rows {
            for j in 1..self.cols {
                if insx[i - 1] || insy[j - 1] {
                    (
                        self.score.m[i][j],
                        self.score.x[i][j],
                        self.score.y[i][j],
                        self.trace.m[i][j],
                    ) = self.insertion_in_either(i, j, &insx, &insy);
                    self.trace.x[i][j] = self.trace.m[i][j];
                    self.trace.y[i][j] = self.trace.m[i][j];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) =
                        self.possible_match(i - 1, j - 1, (setsx[i - 1] & setsy[j - 1]) == 0);
                    (self.score.x[i][j], self.trace.x[i][j]) =
                        self.possible_gap_x(i - 1, j, delx[i - 1]);
                    (self.score.y[i][j], self.trace.y[i][j]) =
                        self.possible_gap_y(i, j - 1, dely[j - 1]);
                }
            }
        }
    }
}

pub(crate) fn pars_align(sets1: &[u8], sets2: &[u8]) -> ParsimonyAlignmentMatrices {
    let rows = sets1.len();
    let cols = sets2.len();
    let d1 = vec![false; rows];
    let i1 = vec![false; rows];
    let d2 = vec![false; cols];
    let i2 = vec![false; cols];
    let a = 2.5;
    let b = 0.5;
    let c = 1.0;

    let mut pars_mats = ParsimonyAlignmentMatrices::new(rows + 1, cols + 1, a, b, c);

    pars_mats.init_x(&i1, &d1);
    pars_mats.init_y(&i2, &d2);
    pars_mats.align_sequences(sets1, sets2, &i1, &d1, &i2, &d2);
    println!("{}", pars_mats);
    pars_mats
}
