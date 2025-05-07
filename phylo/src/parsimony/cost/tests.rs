use crate::alignment::{Alignment, Sequences};
use crate::likelihood::TreeSearchCost;
use crate::parsimony::{BasicParsimonyCost, DolloParsimonyCost};
use crate::phylo_info::PhyloInfo;
use crate::{record_wo_desc as rec, tree};

#[test]
fn basic_parsimony_cost() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
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
    assert_eq!(cost.cost(), -6.0);
}

#[test]
fn basic_parsimony_reroot() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree_reroot = tree!("(((D:1,(B:1,A:0.5):0.5):1,C:2):0);");

    let info_reroot = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree_reroot).unwrap(),
        tree: tree_reroot,
    };
    let cost_reroot = BasicParsimonyCost::new(info_reroot).unwrap();
    assert_eq!(cost_reroot.cost(), -4.0);
    assert_eq!(cost_reroot.cost(), cost.cost());
}

#[test]
fn basic_parsimony_cost_gaps() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GA-T-"),
        rec!("B", b"GA-TT"),
        rec!("C", b"-A-TT"),
        rec!("D", b"-G-TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -3.0);
    assert_eq!(cost.cost(), -3.0);
}

#[test]
fn dollo_parsimony_cost_nogaps() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree2 = tree!("((A:1.0,C:1.0):1.0,(B:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree2).unwrap(),
        tree: tree2,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -5.0);

    let tree3 = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree3).unwrap(),
        tree: tree3,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -6.0);
}

#[test]
fn dollo_parsimony_reroot() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GGA-"),
        rec!("B", b"GGG-"),
        rec!("C", b"ACAG"),
        rec!("D", b"ACGG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree_reroot = tree!("(((D:1,(B:1,A:0.5):0.5):1,C:2):0);");

    let info_reroot = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree_reroot).unwrap(),
        tree: tree_reroot,
    };
    let cost_reroot = DolloParsimonyCost::new(info_reroot).unwrap();
    // This is a non-reversible cost function.
    assert_eq!(cost_reroot.cost(), -5.0);
    assert_ne!(cost_reroot.cost(), cost.cost());
}

#[test]
fn dollo_parsimony_cost_gaps() {
    let seqs = Sequences::new(vec![
        rec!("A", b"GA-T-"),
        rec!("B", b"GA-TT"),
        rec!("C", b"-A-TT"),
        rec!("D", b"-A-TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -1.0);
    assert_eq!(cost.cost(), -1.0);
}

#[test]
fn dollo_parsimony_cost_deletions() {
    let seqs = Sequences::new(vec![
        rec!("A", b"T-"),
        rec!("B", b"TT"),
        rec!("C", b"T-"),
        rec!("D", b"TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -2.0);
}

#[test]
fn dollo_parsimony_cost_low_insertion() {
    let seqs = Sequences::new(vec![
        rec!("A", b"-"),
        rec!("B", b"T"),
        rec!("C", b"-"),
        rec!("D", b"T"),
        rec!("E", b"-"),
        rec!("F", b"-"),
    ]);
    let tree = tree!("(((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):1.0,(E:1.0,F:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = DolloParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -2.0);
}
