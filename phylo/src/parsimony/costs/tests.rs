use crate::alphabets::dna_alphabet;
use crate::parsimony::{costs::ParsimonyCostsSimple, BranchParsimonyCosts};

#[test]
fn default_costs() {
    let def = ParsimonyCostsSimple::new_default();
    assert_eq!(def.branch.r#match(b'A', b'B'), 1.0);
    assert_eq!(def.branch.r#match(b'B', b'A'), 1.0);
    assert_eq!(def.branch.r#match(b'N', b'K'), 1.0);
    assert_eq!(def.branch.r#match(b'C', b'C'), 0.0);
    assert_eq!(def.branch.avg(), 1.0);
    assert_eq!(def.branch.gap_ext(), 0.5);
    assert_eq!(def.branch.gap_open(), 2.5);
}

#[test]
fn simple_costs() {
    let mismatch = 3.0;
    let gap_open = 2.0;
    let gap_ext = 10.5;
    let c = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna_alphabet());
    assert_eq!(c.branch.r#match(b'A', b'B'), mismatch);
    assert_eq!(c.branch.r#match(b'A', b'A'), 0.0);
    assert_eq!(c.branch.avg(), mismatch);
    assert_eq!(c.branch.gap_open(), gap_open * mismatch);
    assert_eq!(c.branch.gap_ext(), gap_ext * mismatch);
    let mismatch = 1.0;
    let gap_open = 10.0;
    let gap_ext = 2.5;
    let c = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna_alphabet());
    assert_eq!(c.branch.r#match(b'A', b'B'), mismatch);
    assert_eq!(c.branch.r#match(b'A', b'A'), 0.0);
    assert_eq!(c.branch.avg(), mismatch);
    assert_eq!(c.branch.gap_open(), gap_open);
    assert_eq!(c.branch.gap_ext(), gap_ext);
}
