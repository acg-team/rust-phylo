pub(crate) type Mapping = Vec<Option<usize>>;

#[derive(Clone, Debug)]
pub(crate) struct Alignment {
    pub(crate) map_x: Mapping,
    pub(crate) map_y: Mapping,
}

impl Alignment {
    pub(super) fn new(x: Mapping, y: Mapping) -> Alignment {
        Alignment { map_x: x, map_y: y }
    }

    pub(super) fn empty() -> Alignment {
        Alignment {
            map_x: vec![],
            map_y: vec![],
        }
    }
}
