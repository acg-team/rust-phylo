use crate::asr::AncestralSequenceReconstruction;
use crate::{
    alignment::{Alignment, Sequences, MASA, MSA},
    parsimony_presence_absence::ParsimonyPresenceAbsence,
    record, tree,
};

#[test]
fn asr_n_seqs_not_same_as_leaves() {
    // arrange
    let tree = tree!("root:1.0;");
    let seqs = Sequences::new(vec![record!("root", Some("seq with 4 nucls"), b"AA--AAA")]);
    let msa = MSA::from_aligned(seqs, &tree).unwrap();
    let wrong_tree =
        tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0) I7:1.0) I8:1.0;");
    let asr = ParsimonyPresenceAbsence {};

    // act
    let error_msg = AncestralSequenceReconstruction::<MSA, MASA>::reconstruct_ancestral_seqs(
        &asr,
        &msa,
        &wrong_tree,
    )
    .unwrap_err()
    .to_string();

    // assert
    assert!(error_msg.contains("but tree has"));
}
