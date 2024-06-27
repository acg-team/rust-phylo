use std::path::PathBuf;

use crate::phylo_info::{GapHandling, PhyloInfo};

#[test]
fn test_full_functionality() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    assert!(info.msa.is_some());

    // let frequencies = info.msa.unwrap();
    // info();

    //     // Optimize GTR model parameters
    //     let gtr_model = GTRModel::optimize(&alignment, &frequencies).unwrap();

    //     // Assert that the model parameters are as expected
    //     // This will depend on your specific test case
    //     assert_eq!(gtr_model.rate_matrix(), &expected_rate_matrix);
    //     assert_eq!(gtr_model.base_frequencies(), &expected_base_frequencies);
}
