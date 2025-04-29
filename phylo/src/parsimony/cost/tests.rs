use crate::alignment::{Alignment, Sequences};
use crate::likelihood::TreeSearchCost;
use crate::parsimony::BasicParsimonyCost;
use crate::phylo_info::PhyloInfo;
use crate::{record_wo_desc as record, tree};

#[test]
fn basic_parsimony_cost() {
    let seqs = Sequences::new(vec![
        record!("A", b"GGA"),
        record!("B", b"GGG"),
        record!("C", b"ACA"),
        record!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree2 = tree!("((A:1.0,C:1.0):1.0,(B:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree2).unwrap(),
        tree: tree2,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -5.0);

    let tree3 = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree3).unwrap(),
        tree: tree3,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), 6.0);
}

#[test]
fn basic_parsimony_cost_gaps() {
    let seqs = Sequences::new(vec![
        record!("A", b"GA-T-"),
        record!("B", b"GA-TT"),
        record!("C", b"-A-TT"),
        record!("D", b"-G-TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -1.0);
    assert_eq!(cost.cost(), -1.0);
}
