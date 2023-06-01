#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParsimonySiteInfo {
    pub(crate) set: u32,
    pub(crate) poss_gap: bool,
    pub(crate) perm_gap: bool,
}

impl ParsimonySiteInfo {
    pub(crate) fn new(set: u32, poss_gap: bool, perm_gap: bool) -> ParsimonySiteInfo {
        ParsimonySiteInfo {
            set,
            poss_gap,
            perm_gap,
        }
    }
    pub(crate) fn new_leaf(set: u32) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(set, false, false)
    }
}