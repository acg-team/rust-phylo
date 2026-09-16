use assert_matches::assert_matches;

use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::PhyloInfoBuilder;
use crate::{record, tree, Error};

#[test]
fn reconstruct_ancestral_seqs_n_seqs_not_same_as_leaves() {
    let tree = tree!("root:1.0;");
    let seqs =
        Sequences::new_unchecked(vec![record!("root", Some("seq with 4 nucls"), b"AA--AAA")]);
    let msa = MSA::from_aligned(seqs, &tree).unwrap();
    let wrong_tree =
        tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0) I7:1.0) I8:1.0;");
    let asr = ParsimonyPresenceAbsence {};

    let err = AncestralSequenceReconstruction::<MSA, MASA>::reconstruct_ancestral_seqs(
        &asr,
        &msa,
        &wrong_tree,
    );

    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("but tree has")
    );
}

#[test]
fn test_reconstruct_ancestral_seqs() {
    let info = PhyloInfoBuilder::with_attrs(
        "./examples/data/sequences_DNA_small.fasta",
        "./examples/data/tree_with_ancestral_ids.newick",
    )
    .build()
    .unwrap();

    let masa: MASA = AncestralSequenceReconstruction::reconstruct_ancestral_seqs(
        &ParsimonyPresenceAbsence {},
        &info.msa,
        &info.tree,
    )
    .unwrap();

    assert_eq!(masa.seq_count(), info.msa.seq_count());
    assert_eq!(
        masa.seq_count() + masa.ancestral_seqs().len(),
        info.tree.len()
    );
    assert_eq!(masa.len(), info.msa.len());
}
