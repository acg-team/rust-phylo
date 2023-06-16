use crate::parsimony_alignment::parsimony_sets::ParsimonySet;
use crate::parsimony_alignment::parsimony_sets::make_parsimony_set;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GapFlag {
    GapFixed,
    GapOpen,
    GapExt,
    NoGap,
}


#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsimonySiteInfo {
    pub(crate) set: ParsimonySet,
    pub(super) flag: GapFlag,
}

impl ParsimonySiteInfo {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, gap_flag: GapFlag) -> ParsimonySiteInfo {
        ParsimonySiteInfo {
            set: make_parsimony_set(set),
            flag: gap_flag,
        }
    }
    pub(crate) fn new_leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(set, GapFlag::NoGap)
    }

    pub(crate) fn is_fixed(&self) -> bool {
        self.flag == GapFlag::GapFixed
    }

    pub(crate) fn is_open(&self) -> bool {
        self.flag == GapFlag::GapOpen
    }

    pub(crate) fn is_ext(&self) -> bool {
        self.flag == GapFlag::GapExt
    }

    pub(crate) fn no_gap(&self) -> bool {
        self.flag == GapFlag::NoGap
    }
}