use crate::parsimony_alignment::parsimony_sets::ParsimonySet;
use crate::parsimony_alignment::parsimony_sets::make_parsimony_set;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GapFlag {
    GapFixed,
    GapPossible,
    NoGap,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsimonySiteInfo {
    pub(crate) set: ParsimonySet,
    pub(crate) gap_flag: GapFlag,
}

impl ParsimonySiteInfo {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, gap_flag: GapFlag) -> ParsimonySiteInfo {
        ParsimonySiteInfo {
            set: make_parsimony_set(set),
            gap_flag,
        }
    }
    pub(crate) fn new_leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(set, GapFlag::NoGap)
    }
}