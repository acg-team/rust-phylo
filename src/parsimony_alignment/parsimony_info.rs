use std::fmt;
use std::fmt::Debug;

use crate::parsimony_alignment::parsimony_sets::make_parsimony_set;
use crate::parsimony_alignment::parsimony_sets::ParsimonySet;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SiteFlag {
    GapFixed,
    GapOpen,
    GapExt,
    NoGap,
}

#[derive(Clone, PartialEq)]
pub(crate) struct ParsimonySiteInfo {
    pub(crate) set: ParsimonySet,
    pub(super) flag: SiteFlag,
}

impl Debug for ParsimonySiteInfo {
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

impl ParsimonySiteInfo {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, gap_flag: SiteFlag) -> ParsimonySiteInfo {
        ParsimonySiteInfo {
            set: make_parsimony_set(set),
            flag: gap_flag,
        }
    }
    pub(crate) fn new_leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(set, SiteFlag::NoGap)
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
