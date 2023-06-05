use crate::parsimony_alignment::parsimony_sets::ParsimonySet;
use crate::parsimony_alignment::parsimony_sets::make_parsimony_set;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsimonySiteInfo {
    pub(crate) set: ParsimonySet,
    pub(crate) poss_gap: bool,
    pub(crate) perm_gap: bool,
}

impl ParsimonySiteInfo {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, poss_gap: bool, perm_gap: bool) -> ParsimonySiteInfo {
        ParsimonySiteInfo {
            set: make_parsimony_set(set),
            poss_gap,
            perm_gap,
        }
    }
    pub(crate) fn new_leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(set, false, false)
    }
}