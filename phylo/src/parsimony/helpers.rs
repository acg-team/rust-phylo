use std::fmt::{self, Debug};

use rand::random;

use crate::alphabets::ParsimonySet;

#[derive(Clone, Debug, PartialEq)]
pub struct Rounding {
    is_round: bool,
    pub digits: usize,
}

impl Rounding {
    pub fn zero() -> Self {
        Rounding {
            is_round: true,
            digits: 0,
        }
    }
    pub fn four() -> Self {
        Rounding {
            is_round: true,
            digits: 4,
        }
    }
    pub fn none() -> Self {
        Rounding {
            is_round: false,
            digits: 0,
        }
    }
    pub fn yes(&self) -> bool {
        self.is_round
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalZeros {
    is_zero: bool,
}

impl DiagonalZeros {
    pub fn zero() -> Self {
        DiagonalZeros { is_zero: true }
    }
    pub fn non_zero() -> Self {
        DiagonalZeros { is_zero: false }
    }
    pub fn yes(&self) -> bool {
        self.is_zero
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapInY,
    GapInX,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SiteFlag {
    GapFixed,
    GapOpen,
    GapExt,
    NoGap,
}

#[derive(Clone, PartialEq)]
pub(crate) struct ParsimonySite {
    pub(crate) set: ParsimonySet,
    pub(super) flag: SiteFlag,
}

impl Debug for ParsimonySite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}: {:?}",
            self.set.iter().map(|&a| a as char).collect::<Vec<char>>(),
            self.flag
        )
        .unwrap();
        Ok(())
    }
}

impl ParsimonySite {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, gap_flag: SiteFlag) -> ParsimonySite {
        ParsimonySite {
            set: ParsimonySet::from_iter(set),
            flag: gap_flag,
        }
    }
    pub(crate) fn leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySite {
        ParsimonySite::new(set, SiteFlag::NoGap)
    }

    pub(crate) fn is_fixed(&self) -> bool {
        self.flag == SiteFlag::GapFixed
    }

    #[allow(dead_code)]
    pub(crate) fn is_open(&self) -> bool {
        self.flag == SiteFlag::GapOpen
    }

    pub(crate) fn is_ext(&self) -> bool {
        self.flag == SiteFlag::GapExt
    }

    pub(crate) fn is_possible(&self) -> bool {
        self.flag == SiteFlag::GapOpen || self.flag == SiteFlag::GapExt
    }

    pub(crate) fn no_gap(&self) -> bool {
        self.flag == SiteFlag::NoGap
    }
}

pub(crate) fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}
