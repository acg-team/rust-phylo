use approx::relative_eq;

use super::*;

impl tree::Node {
    fn new(idx: usize, parent: Option<usize>, children: Vec<usize>, blength: f32) -> Self {
        Self {
            idx: idx,
            parent: parent,
            children: children,
            blen: blength,
        }
    }
}

impl PartialEq for tree::Node {
    fn eq(&self, other: &Self) -> bool {
        (self.idx == other.idx)
            && (self.parent == other.parent)
            && (self.children.iter().min() == other.children.iter().min())
            && (self.children.iter().max() == other.children.iter().max())
            && relative_eq!(self.blen, other.blen)
    }
}

#[test]
fn nj_correct() {
    let nj_distances = njmat::NJMat {
        idx: (0..5).collect(),
        distances: dmatrix![
            0.0, 5.0, 9.0, 9.0, 8.0;
            5.0, 0.0, 10.0, 10.0, 9.0;
            9.0, 10.0, 0.0, 8.0, 7.0;
            9.0, 10.0, 8.0, 0.0, 3.0;
            8.0, 9.0, 7.0, 3.0, 0.0],
    };

    let nj_tree = tree::build_nj_tree(nj_distances).unwrap();

    let result = vec![
        tree::Node::new(0, Some(5), Vec::new(), 2.0),
        tree::Node::new(1, Some(5), Vec::new(), 3.0),
        tree::Node::new(2, Some(6), Vec::new(), 4.0),
        tree::Node::new(3, Some(7), Vec::new(), 2.0),
        tree::Node::new(4, Some(7), Vec::new(), 1.0),
        tree::Node::new(5, Some(6), vec![1, 0], 3.0),
        tree::Node::new(6, Some(8), vec![2, 5], 1.0),
        tree::Node::new(7, Some(8), vec![4, 3], 1.0),
        tree::Node::new(8, None, vec![7, 6], 0.0),
    ];

    assert_eq!(nj_tree.root, 8);
    for (i, node) in nj_tree.nodes.into_iter().enumerate() {
        assert_eq!(node, result[i]);
    }
}
