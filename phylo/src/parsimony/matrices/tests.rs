use approx::assert_relative_eq;

use crate::alphabets::dna_alphabet as dna;
use crate::parsimony::costs::parsimony_costs_model::ParsimonyCostsWModel;
use crate::parsimony::costs::{GapMultipliers, ParsimonyCostsSimple};
use crate::parsimony::matrices::Direction::{GapInX, GapInY, Matc};
use crate::parsimony::ParsimonySite as PSI;
use crate::parsimony::SiteFlag::{GapExt, GapFixed, GapOpen, NoGap};
use crate::parsimony::{ParsimonyAlignmentMatrices as PAM, Rounding};
use crate::substitution_models::{SubstModel, K80};

macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

#[test]
fn fill_matrix() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.5;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost, dna());

    let node_info_1 = vec![PSI::new([b'C'], NoGap), PSI::new([b'C'], NoGap)];
    let node_info_2 = vec![PSI::new([b'A'], NoGap), PSI::new([b'C'], NoGap)];

    let mut pars_mats = PAM::new(&node_info_1, 1.0, &node_info_2, 1.0, &scoring, |_| 0);

    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf],
            vec![inf, 1.0, 2.5],
            vec![inf, 3.5, 1.0]
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf],
            vec![2.5, 5.0, 5.5],
            vec![3.0, 3.5, 5.0]
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 2.5, 3.0],
            vec![inf, 5.0, 3.5],
            vec![inf, 5.5, 6.0]
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![GapInY, Matc, GapInX],
            vec![GapInY, GapInY, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, Matc, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc]
        ]
    );
}

#[test]
fn fill_matrix_other_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.5;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let node_info_1 = vec![PSI::new([b'C'], NoGap), PSI::new([b'C'], NoGap)];
    let node_info_2 = vec![PSI::new([b'A'], NoGap), PSI::new([b'C'], NoGap)];

    let mut pars_mats = PAM::new(&node_info_1, 1.0, &node_info_2, 1.0, &scoring, |l| l - 1);

    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf],
            vec![inf, 1.0, 2.5],
            vec![inf, 3.5, 1.0]
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf],
            vec![2.5, 5.0, 5.5],
            vec![3.0, 3.5, 5.0]
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 2.5, 3.0],
            vec![inf, 5.0, 3.5],
            vec![inf, 5.5, 6.0]
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![GapInY, GapInY, GapInX],
            vec![GapInY, GapInY, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, Matc, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY]
        ]
    );
}

#[test]
fn traceback_correct() {
    let mismatch = 1.0;
    let gap_open = 2.5;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let node_info_1 = vec![PSI::new([b'C'], NoGap), PSI::new([b'C'], NoGap)];
    let node_info_2 = vec![PSI::new([b'A'], NoGap), PSI::new([b'C'], NoGap)];

    let mut pars_mats = PAM::new(&node_info_1, 1.0, &node_info_2, 1.0, &scoring, |l| l - 1);

    pars_mats.fill_matrices();

    let (node_info, alignment, score) = pars_mats.traceback();
    assert_eq!(node_info[0], PSI::new([b'C', b'A'], NoGap));
    assert_eq!(node_info[1], PSI::new([b'C'], NoGap));
    assert_eq!(alignment.map_x, align!(0 1));
    assert_eq!(alignment.map_y, align!(0 1));
    assert_eq!(score, 1.0);

    let mut pars_mats = PAM::new(&node_info_1, 1.0, &node_info_2, 1.0, &scoring, |_| 0);

    pars_mats.fill_matrices();

    let (node_info, alignment, score) = pars_mats.traceback();
    assert_eq!(node_info[0], PSI::new([b'C', b'A'], NoGap));
    assert_eq!(node_info[1], PSI::new([b'C'], NoGap));
    assert_eq!(alignment.map_x, align!(0 1));
    assert_eq!(alignment.map_y, align!(0 1));
    assert_eq!(score, 1.0);
}

#[test]
fn fill_matrix_gap_adjustment_1() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_1.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_1.newick
    let mismatch = 1.0;
    let gap_open = 5.5;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let left_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'A'], GapExt),
    ];
    let right_info = vec![PSI::new([b'A', b'C'], NoGap), PSI::new([b'C', b'A'], NoGap)];

    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf],
            vec![inf, 0.0, 5.5],
            vec![inf, 5.5, 0.0],
            vec![inf, 6.0, 5.5],
            vec![inf, 11.0, 6.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf],
            vec![5.5, 11.0, 11.5],
            vec![6.0, 5.5, 11.0],
            vec![6.0, 5.5, 0.0],
            vec![6.0, 5.5, 0.0],
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 5.5, 6.0],
            vec![inf, 11.0, 5.5],
            vec![inf, 11.5, 11.0],
            vec![inf, 11.5, 11.0],
            vec![inf, 11.5, 11.0],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![GapInY, Matc, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, Matc, Matc],
            vec![GapInY, Matc, Matc],
            vec![GapInY, GapInY, GapInY],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
        ]
    );
}

#[test]
fn traceback_gap_adjustment_1() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_1.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_1.newick
    let mismatch = 1.0;
    let gap_open = 5.5;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let left_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'A'], GapExt),
    ];
    let right_info = vec![PSI::new([b'A', b'C'], NoGap), PSI::new([b'C', b'A'], NoGap)];

    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();
    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
    assert_eq!(score + 8.0, 8.0);
}

#[test]
fn fill_matrix_gap_adjustment_2() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_2.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_2.newick
    let mismatch = 1.0;
    let gap_open = 4.5;
    let gap_ext = 1.0;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let sites_left = vec![
        PSI::new([b'A'], GapOpen),
        PSI::new([b'C'], GapExt),
        PSI::new([b'G', b'C'], NoGap),
    ];
    let sites_right = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'G'], GapExt),
    ];
    let mut pars_mats = PAM::new(&sites_left, 1.0, &sites_right, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf, inf],
            vec![inf, 0.0, 5.5, 9.0],
            vec![inf, 4.5, 0.0, 1.0],
            vec![inf, 1.0, 3.5, 0.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf, inf],
            vec![0.0, 4.5, 4.5, 4.5],
            vec![0.0, 3.5, 3.5, 3.5],
            vec![4.5, 8.0, 4.5, 5.5],
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 4.5, 4.5, 4.5],
            vec![inf, 4.5, 0.0, 0.0],
            vec![inf, 4.5, 3.5, 3.5],
            vec![inf, 9.0, 1.0, 1.0],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX, GapInX],
            vec![GapInY, Matc, GapInX, GapInX],
            vec![GapInY, GapInY, Matc, GapInX],
            vec![GapInY, GapInY, GapInY, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX, GapInX],
            vec![GapInY, Matc, GapInX, GapInX],
            vec![GapInY, GapInY, Matc, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc, GapInX],
            vec![GapInY, GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc, GapInX],
        ]
    );
}

#[test]
fn traceback_gap_adjustment_2() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_2.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_2.newick
    let mismatch = 1.0;
    let gap_open = 4.5;
    let gap_ext = 1.0;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let sites_left = vec![
        PSI::new([b'A'], GapOpen),
        PSI::new([b'C'], GapExt),
        PSI::new([b'G', b'C'], NoGap),
    ];
    let sites_right = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'G'], GapExt),
    ];
    let mut pars_mats = PAM::new(&sites_left, 1.0, &sites_right, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();
    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'G'], NoGap),
    ];
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2));
    assert_eq!(alignment.map_y, align!(0 1 2));
    assert_eq!(score + 12.0, 12.0);
}

#[test]
fn fill_matrix_gap_adjustment_3() {
    // Sequence file: sequences_fill_matrix_gap_adjustment_3.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_3.newick
    let mismatch = 1.0;
    let gap_open = 0.75;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let left_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |l| l - 1);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf],
            vec![inf, inf, inf],
            vec![inf, 1.0, 0.75],
            vec![inf, 1.0, 0.75],
            vec![inf, 0.75, 2.0],
            vec![inf, 0.75, 2.0],
            vec![inf, 0.75, 2.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf],
            vec![0.0, inf, inf],
            vec![0.0, 0.75, 1.25],
            vec![0.75, 1.5, 1.5],
            vec![1.25, 1.75, 1.5],
            vec![1.25, 1.75, 1.5],
            vec![1.25, 1.75, 1.5],
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 0.75, 1.25],
            vec![inf, 0.75, 1.25],
            vec![inf, 0.75, 1.25],
            vec![inf, 1.5, 1.75],
            vec![inf, 2.0, 1.5],
            vec![inf, 2.0, 1.5],
            vec![inf, 2.0, 1.5],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![Matc, Matc, Matc],
            vec![GapInY, GapInY, GapInX],
            vec![GapInY, GapInY, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![Matc, Matc, Matc],
            vec![Matc, Matc, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, Matc],
            vec![GapInY, Matc, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
        ]
    );
}

#[test]
fn traceback_gap_adjustment_3() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_3.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_3.newick
    let mismatch = 1.0;
    let gap_open = 0.75;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let left_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |l| l - 1);

    pars_mats.fill_matrices();
    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'C'], NoGap),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    assert_eq!(node_info.len(), true_info.len());
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2 3 - 4 5));
    assert_eq!(alignment.map_y, align!(- - - 0 1 - -));
    assert_eq!(score + 2.75, 4.25);
}

#[test]
fn fill_matrix_gap_adjustment_4() {
    // Sequence file: sequences_fill_matrix_gap_adjustment_3.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_3.newick
    // Slightly different setup to ensure there's a gap opening at the beginning of the alignment
    let mismatch = 1.0;
    let gap_open = 0.75;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let left_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'C'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, inf, inf],
            vec![inf, inf, inf],
            vec![inf, 1.0, 0.75],
            vec![inf, 1.75, 1.0],
            vec![inf, 0.75, 2.0],
            vec![inf, 0.75, 2.0],
            vec![inf, 0.75, 2.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, inf, inf],
            vec![0.0, inf, inf],
            vec![0.75, 1.5, 2.0],
            vec![0.75, 1.0, 0.75],
            vec![1.25, 1.75, 1.5],
            vec![1.25, 1.75, 1.5],
            vec![1.25, 1.75, 1.5],
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 0.75, 1.25],
            vec![inf, 0.75, 1.25],
            vec![inf, 1.5, 1.75],
            vec![inf, 1.5, 1.75],
            vec![inf, 2.0, 1.5],
            vec![inf, 2.0, 1.5],
            vec![inf, 2.0, 1.5],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![Matc, Matc, Matc],
            vec![GapInY, GapInY, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![Matc, Matc, Matc],
            vec![Matc, Matc, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, Matc, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, Matc],
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
        ]
    );
}

#[test]
fn traceback_gap_adjustment_4() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_3.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_3.newick
    // Slightly different setup to ensure there's a gap opening at the beginning of the alignment
    let mismatch = 1.0;
    let gap_open = 0.75;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let left_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'A'], GapOpen),
        PSI::new([b'C'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 1.0, &scoring, |_| 0);
    pars_mats.fill_matrices();
    let (node_info, alignment, score) = pars_mats.traceback();

    let true_info = vec![
        PSI::new([b'C'], GapOpen),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
    ];
    assert_eq!(node_info.len(), true_info.len());
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(- 0 1 2 3 4 5));
    assert_eq!(alignment.map_y, align!(0 - 1 - - - -));
    assert_eq!(score + 2.75, 4.25);
}

#[cfg(test)]
fn assert_float_relative_matrix_eq(actual: &[Vec<f64>], expected: &[Vec<f64>], epsilon: f64) {
    use approx::assert_relative_eq;

    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(
            actual_row.as_slice(),
            expected_row.as_slice(),
            epsilon = epsilon
        );
    }
}

#[test]
fn fill_matrix_diff_branch_models() {
    // Sequence file: sequences_diff_branch_lengths_1.fasta
    // Tree file: tree_diff_branch_lengths_1.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[1.0, 2.0],
        false,
        &GapMultipliers {
            open: 2.0,
            ext: 0.5,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'A'], NoGap),
    ];
    let right_info = vec![PSI::new([b'A'], NoGap), PSI::new([b'C'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 2.0, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    let expected_m = vec![
        vec![0.0, inf, inf],
        vec![inf, 1.966, 6.472],
        vec![inf, 5.90805, 3.932],
        vec![inf, 7.3974875, 7.51765],
        vec![inf, 8.530525, 9.3634875],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.m, &expected_m, 0.001);
    let expected_x = vec![
        vec![0.0, inf, inf],
        vec![3.58565, 7.73525, 9.654125],
        vec![5.0750875, 5.55165, 9.70125],
        vec![6.564525, 7.0410875, 7.51765],
        vec![8.0539625, 8.530525, 9.0070875],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.x, &expected_x, 0.001);
    let expected_y = vec![
        vec![0.0, 4.1496, 6.068475],
        vec![inf, 7.73525, 6.1156],
        vec![inf, 9.2246875, 9.70125],
        vec![inf, 10.714125, 11.1906875],
        vec![inf, 12.2035625, 12.680125],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.y, &expected_y, 0.001);
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![GapInY, Matc, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, Matc, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY]
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY]
        ]
    );
}

#[test]
fn traceback_diff_branch_models() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_diff_branch_lengths_1.fasta
    // Tree file: tree_diff_branch_lengths_1.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[1.0, 2.0],
        false,
        &GapMultipliers {
            open: 2.0,
            ext: 0.5,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'A'], NoGap),
    ];
    let right_info = vec![PSI::new([b'A'], NoGap), PSI::new([b'C'], NoGap)];
    let mut pars_mats = PAM::new(&left_info, 1.0, &right_info, 2.0, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
        PSI::new([b'C'], GapOpen),
        PSI::new([b'A'], GapExt),
    ];
    assert_eq!(node_info.len(), true_info.len());
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
    assert_relative_eq!(score, 9.0070875, epsilon = 0.0001);
}

#[test]
fn fill_matrix_diff_branch_models_2() {
    // Sequence file: sequences_diff_branch_lengths_2.fasta
    // Tree file: tree_diff_branch_lengths_2.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[3.5, 3.0],
        false,
        &GapMultipliers {
            open: 1.5,
            ext: 0.75,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'G'], GapOpen),
        PSI::new([b'A'], GapExt),
        PSI::new([b'C'], GapExt),
        PSI::new([b'G', b'A'], NoGap),
        PSI::new([b'C', b'G'], NoGap),
    ];
    let right_info = vec![
        PSI::new([b'A', b'C'], NoGap),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], GapOpen),
    ];

    let mut pars_mats = PAM::new(&left_info, 3.0, &right_info, 3.5, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    let expected_m = vec![
        vec![0.0, inf, inf, inf],
        vec![inf, 2.6836, 6.1115125, 8.54886875],
        vec![inf, 3.7033, 5.3468, 8.8453125],
        vec![inf, 3.7033, 6.4371, 8.01],
        vec![inf, 2.6632, 6.0911125, 8.54886875],
        vec![inf, 6.0602, 5.3468, 8.7543125],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.m, &expected_m, 0.001);
    let expected_x = vec![
        vec![0.0, inf, inf, inf],
        vec![0.0, 3.4279125, 5.81506875, 5.81506875],
        vec![0.0, 3.4279125, 5.81506875, 5.81506875],
        vec![0.0, 3.4279125, 5.81506875, 5.81506875],
        vec![3.397, 6.8249125, 9.21206875, 9.21206875],
        vec![5.7539, 6.0602, 9.4881125, 9.4881125],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.x, &expected_x, 0.001);
    let expected_y = vec![
        vec![0.0, 3.4279125, 5.81506875, 5.81506875],
        vec![inf, 3.4279125, 5.81506875, 5.81506875],
        vec![inf, 3.4279125, 5.81506875, 5.3468],
        vec![inf, 3.4279125, 5.81506875, 5.81506875],
        vec![inf, 6.8249125, 6.0911125, 6.0911125],
        vec![inf, 9.1818125, 9.4881125, 5.3468],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.y, &expected_y, 0.001);
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX, GapInX],
            vec![GapInY, Matc, GapInX, GapInX],
            vec![GapInY, GapInY, Matc, Matc],
            vec![GapInY, GapInY, Matc, Matc],
            vec![GapInY, GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, Matc, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY, GapInY],
            vec![GapInY, Matc, Matc, GapInX],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, GapInX, GapInY],
            vec![GapInY, GapInY, GapInX, Matc],
            vec![GapInY, GapInY, GapInX, GapInY],
            vec![GapInY, GapInY, Matc, Matc],
            vec![GapInY, GapInY, Matc, Matc],
        ]
    );
}

#[test]
fn traceback_diff_branch_models_2() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_diff_branch_lengths_2.fasta
    // Tree file: tree_diff_branch_lengths_2.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[3.5, 3.0],
        false,
        &GapMultipliers {
            open: 1.5,
            ext: 0.75,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'G'], GapOpen),
        PSI::new([b'A'], GapExt),
        PSI::new([b'C'], GapExt),
        PSI::new([b'G', b'A'], NoGap),
        PSI::new([b'C', b'G'], NoGap),
    ];
    let right_info = vec![
        PSI::new([b'A', b'C'], NoGap),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], GapOpen),
    ];

    let mut pars_mats = PAM::new(&left_info, 3.0, &right_info, 3.5, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C', b'G', b'A'], NoGap),
        PSI::new([b'-'], GapFixed),
    ];
    assert_eq!(node_info.len(), true_info.len());
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2 3 4 -));
    assert_eq!(alignment.map_y, align!(- - - 0 1 2));
    assert_relative_eq!(score + 7.921925 + 11.028825, 24.29755, epsilon = 0.0001);
}

#[test]
fn fill_matrix_diff_branch_models_3() {
    // Sequence file: sequences_diff_branch_lengths_3.fasta
    // Tree file: tree_diff_branch_lengths_3.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[0.52, 2.58],
        false,
        &GapMultipliers {
            open: 1.0,
            ext: 0.75,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'A'], GapOpen),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];

    let mut pars_mats = PAM::new(&left_info, 0.6, &right_info, 2.6, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let inf = f64::INFINITY;
    let expected_m = vec![
        vec![0.0, inf, inf],
        vec![inf, 1.9306, 4.7206],
        vec![inf, 1.9306, 4.7206],
        vec![inf, 1.9306, 4.7206],
        vec![inf, 1.9306, 4.7206],
        vec![inf, 1.9306, 3.6713],
        vec![inf, 3.59575, 3.8612],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.m, &expected_m, 0.001);
    let expected_x = vec![
        vec![0.0, inf, inf],
        vec![0.0, 2.9799, 5.533625],
        vec![0.0, 2.9799, 5.533625],
        vec![0.0, 2.9799, 5.533625],
        vec![0.0, 2.9799, 5.533625],
        vec![1.85505, 3.78565, 6.57565],
        vec![3.3627125, 3.78565, 5.52635],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.x, &expected_x, 0.001);
    let expected_y = vec![
        vec![0.0, 2.9799, 5.533625],
        vec![inf, 2.9799, 4.9105],
        vec![inf, 2.9799, 4.9105],
        vec![inf, 2.9799, 4.9105],
        vec![inf, 2.9799, 4.9105],
        vec![inf, 4.83495, 4.9105],
        vec![inf, 6.3426125, 6.57565],
    ];
    assert_float_relative_matrix_eq(&pars_mats.score.y, &expected_y, 0.001);
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Matc, GapInX, GapInX],
            vec![GapInY, Matc, GapInX],
            vec![Matc, Matc, Matc],
            vec![Matc, Matc, Matc],
            vec![Matc, Matc, Matc],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInX, GapInX],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, GapInY, GapInY],
            vec![GapInY, Matc, Matc],
            vec![GapInY, Matc, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
            vec![GapInX, GapInX, GapInX],
            vec![GapInY, GapInY, Matc],
            vec![GapInY, GapInY, Matc],
        ]
    );
}

#[test]
fn traceback_diff_branch_models_3() {
    // Sequence file: sequences_diff_branch_lengths_3.fasta
    // Tree file: tree_diff_branch_lengths_3.newick
    let model = SubstModel::<K80>::new(&[], &[]);
    let scoring = ParsimonyCostsWModel::new(
        model,
        &[0.52, 2.58],
        false,
        &GapMultipliers {
            open: 1.0,
            ext: 0.75,
        },
        &Rounding::none(),
    )
    .unwrap();

    let left_info = vec![
        PSI::new([b'A'], GapOpen),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'A'], NoGap),
        PSI::new([b'C'], NoGap),
    ];
    let right_info = vec![PSI::new([b'C'], NoGap), PSI::new([b'A'], NoGap)];

    let mut pars_mats = PAM::new(&left_info, 0.6, &right_info, 2.6, &scoring, |_| 0);
    pars_mats.fill_matrices();

    let (node_info, alignment, score) = pars_mats.traceback();
    let true_info = vec![
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'-'], GapFixed),
        PSI::new([b'C', b'A'], NoGap),
        PSI::new([b'A', b'C'], NoGap),
    ];
    assert_eq!(node_info.len(), true_info.len());
    assert_eq!(node_info, true_info);
    assert_eq!(alignment.map_x, align!(0 1 2 3 4 5));
    assert_eq!(alignment.map_y, align!(- - - - 0 1));
    assert_relative_eq!(score + 7.686975 + 8.619275, 20.16745, epsilon = 0.0001);
}
