use crate::parsimony_alignment::{
    parsimony_costs::{parsimony_costs_simple::ParsimonyCostsSimple, ParsimonyCosts},
    parsimony_info::{
        GapFlag::{GapExt, GapFixed, GapOpen, NoGap},
        ParsimonySiteInfo,
    },
    parsimony_matrices::ParsimonyAlignmentMatrices,
    Direction::{GapX, GapY, Matc, Skip},
};

use std::f64::INFINITY as INF;

#[test]
fn fill_matrix() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.5;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];

    let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |_| 0);

    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );

    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, INF, INF],
            vec![INF, 1.0, 2.5],
            vec![INF, 3.5, 1.0]
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, INF, INF],
            vec![2.5, 5.0, 5.5],
            vec![3.0, 3.5, 5.0]
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 2.5, 3.0],
            vec![INF, 5.0, 3.5],
            vec![INF, 5.5, 6.0]
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, Matc, GapY],
            vec![GapX, GapX, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, Matc, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, Matc]
        ]
    );
}

#[test]
fn fill_matrix_other_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.5;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];

    let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |l| l - 1);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );

    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, INF, INF],
            vec![INF, 1.0, 2.5],
            vec![INF, 3.5, 1.0]
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, INF, INF],
            vec![2.5, 5.0, 5.5],
            vec![3.0, 3.5, 5.0]
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 2.5, 3.0],
            vec![INF, 5.0, 3.5],
            vec![INF, 5.5, 6.0]
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, GapX, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, Matc, Matc]
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, GapY]
        ]
    );
}

#[test]
fn traceback_correct() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.5;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
    ];
    let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |l| l - 1);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );

    let (node_info, alignment, score) = pars_mats.traceback(&node_info_1, &node_info_2);
    assert_eq!(node_info[0], ParsimonySiteInfo::new([b'C', b'A'], NoGap));
    assert_eq!(node_info[1], ParsimonySiteInfo::new([b'C'], NoGap));
    assert_eq!(alignment.map_x, vec![Some(0), Some(1)]);
    assert_eq!(alignment.map_y, vec![Some(0), Some(1)]);
    assert_eq!(score, 1.0);

    let mut pars_mats = ParsimonyAlignmentMatrices::new(3, 3, |_| 0);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );

    let (node_info, alignment, score) = pars_mats.traceback(&node_info_1, &node_info_2);
    assert_eq!(node_info[0], ParsimonySiteInfo::new([b'C', b'A'], NoGap));
    assert_eq!(node_info[1], ParsimonySiteInfo::new([b'C'], NoGap));
    assert_eq!(alignment.map_x, vec![Some(0), Some(1)]);
    assert_eq!(alignment.map_y, vec![Some(0), Some(1)]);
    assert_eq!(score, 1.0);
}

#[test]
fn fill_matrix_gap_adjustment_1() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_1.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_1.newick
    let mismatch_cost = 1.0;
    let gap_open_cost = 5.5;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);
    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'C'], GapOpen),
        ParsimonySiteInfo::new([b'A'], GapExt),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'A', b'C'], NoGap),
        ParsimonySiteInfo::new([b'C', b'A'], NoGap),
    ];
    let mut pars_mats = ParsimonyAlignmentMatrices::new(5, 3, |_| 0);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, INF, INF],
            vec![INF, 0.0, 5.5],
            vec![INF, 5.5, 0.0],
            vec![INF, 6.0, 5.5],
            vec![INF, 11.0, 6.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, INF, INF],
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
            vec![INF, 11.0, 5.5],
            vec![INF, 11.5, 11.0],
            vec![INF, 11.5, 11.0],
            vec![INF, 11.5, 11.0],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, Matc, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, Matc, Matc],
            vec![GapX, Matc, Matc],
            vec![GapX, GapX, GapX],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, GapX],
            vec![GapX, GapX, GapX],
        ]
    );
}

#[test]
fn fill_matrix_gap_adjustment_2() {
    // Last step of the alignment with gap adjustments
    // Sequence file: sequences_fill_matrix_gap_adjustment_2.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_2.newick
    let mismatch_cost = 1.0;
    let gap_open_cost = 4.5;
    let gap_ext_cost = 1.0;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);
    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'A'], GapOpen),
        ParsimonySiteInfo::new([b'C'], GapExt),
        ParsimonySiteInfo::new([b'G', b'C'], NoGap),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], GapOpen),
        ParsimonySiteInfo::new([b'G'], GapExt),
    ];
    let mut pars_mats = ParsimonyAlignmentMatrices::new(4, 4, |_| 0);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, INF, INF, INF],
            vec![INF, 0.0, 5.5, 9.0],
            vec![INF, 4.5, 0.0, 1.0],
            vec![INF, 1.0, 3.5, 0.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, INF, INF, INF],
            vec![0.0, 4.5, 4.5, 4.5],
            vec![0.0, 3.5, 3.5, 3.5],
            vec![4.5, 8.0, 4.5, 5.5],
        ]
    );
    assert_eq!(
        pars_mats.score.y,
        vec![
            vec![0.0, 4.5, 4.5, 4.5],
            vec![INF, 4.5, 0.0, 0.0],
            vec![INF, 4.5, 3.5, 3.5],
            vec![INF, 9.0, 1.0, 1.0],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Skip, GapY, GapY, GapY],
            vec![GapX, Matc, GapY, GapY],
            vec![GapX, GapX, Matc, GapY],
            vec![GapX, GapX, GapX, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![Skip, GapY, GapY, GapY],
            vec![GapX, GapY, GapY, GapY],
            vec![GapX, Matc, GapY, GapY],
            vec![GapX, GapX, Matc, Matc],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![Skip, GapY, GapY, GapY],
            vec![GapX, GapX, Matc, GapY],
            vec![GapX, GapX, GapX, Matc],
            vec![GapX, GapX, Matc, GapY],
        ]
    );
}

#[test]
fn fill_matrix_gap_adjustment_3() {
    // Last step of the alignment with some fixed gaps
    // Sequence file: sequences_fill_matrix_gap_adjustment_3.fasta
    // Tree file: tree_fill_matrix_gap_adjustment_3.newick
    let mismatch_cost = 1.0;
    let gap_open_cost = 0.75;
    let gap_ext_cost = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);
    let node_info_1 = vec![
        ParsimonySiteInfo::new([b'-'], GapFixed),
        ParsimonySiteInfo::new([b'A'], GapOpen),
        ParsimonySiteInfo::new([b'A'], NoGap),
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'-'], GapFixed),
        ParsimonySiteInfo::new([b'-'], GapFixed),
    ];
    let node_info_2 = vec![
        ParsimonySiteInfo::new([b'C'], NoGap),
        ParsimonySiteInfo::new([b'A'], NoGap),
    ];
    let mut pars_mats = ParsimonyAlignmentMatrices::new(7, 3, |_| 0);
    pars_mats.fill_matrices(
        &node_info_1,
        &scoring.get_branch_costs(1.0),
        &node_info_2,
        &scoring.get_branch_costs(1.0),
    );
    assert_eq!(
        pars_mats.score.m,
        vec![
            vec![0.0, INF, INF],
            vec![INF, INF, INF],
            vec![INF, 1.0, 0.75],
            vec![INF, 1.0, 0.75],
            vec![INF, 0.75, 2.0],
            vec![INF, 0.75, 2.0],
            vec![INF, 0.75, 2.0],
        ]
    );
    assert_eq!(
        pars_mats.score.x,
        vec![
            vec![0.0, INF, INF],
            vec![0.0, INF, INF],
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
            vec![INF, 0.75, 1.25],
            vec![INF, 0.75, 1.25],
            vec![INF, 1.5, 1.75],
            vec![INF, 2.0, 1.5],
            vec![INF, 2.0, 1.5],
            vec![INF, 2.0, 1.5],
        ]
    );
    assert_eq!(
        pars_mats.trace.m,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, GapX, GapY],
            vec![GapX, GapX, GapX], // vec![GapX, GapX, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapY, GapY],
            vec![GapX, GapY, GapY],
        ]
    );
    assert_eq!(
        pars_mats.trace.x,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, GapX, Matc], // vec![GapX, GapY, Matc],
            vec![GapX, Matc, Matc],
            vec![GapX, GapY, GapY],
            vec![GapX, GapY, GapY],
        ]
    );
    assert_eq!(
        pars_mats.trace.y,
        vec![
            vec![Skip, GapY, GapY],
            vec![GapX, GapY, GapY],
            vec![GapX, GapX, GapY],
            vec![GapX, GapX, Matc],
            vec![GapX, GapX, Matc],
            vec![GapX, GapY, GapY],
            vec![GapX, GapY, GapY],
        ]
    );
}
