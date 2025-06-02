use std::path::{Path, PathBuf};

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences};
use crate::alphabets::{protein_alphabet, AMINOACIDS as aas, GAP, NUCLEOTIDES as nucls};
use crate::evolutionary_models::EvoModel;
use crate::io::read_sequences;
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCostBuilder as PIPB, PIPModel, PIPModelInfo};
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, QMatrixMaker, SubstMatrix, SubstModel,
};

use crate::{frequencies, record_wo_desc as record, tree};

const UNNORMALIZED_PIP_HKY_Q: [f64; 25] = [
    -0.9, 0.11, 0.22, 0.22, 0.0, 0.13, -0.88, 0.26, 0.26, 0.0, 0.33, 0.33, -0.825, 0.165, 0.0,
    0.19, 0.19, 0.095, -0.895, 0.0, 0.25, 0.25, 0.25, 0.25, -0.0,
];

#[cfg(test)]
fn compare_pip_subst_rates_template<Q: QMatrix + QMatrixMaker>(chars: &[u8]) {
    let pip_model = PIPModel::<Q>::new(&[], &[0.1, 0.4]);
    let subst_model = SubstModel::<Q>::new(&[], &[]);
    for (i, &char) in chars.iter().enumerate() {
        assert!(pip_model.rate(char, char) < 0.0);
        assert_relative_eq!(pip_model.q.row(i).sum(), 0.0, epsilon = 1e-10);
        for &other_char in chars {
            if char == other_char {
                continue;
            }
            assert_relative_eq!(
                pip_model.rate(char, other_char),
                subst_model.rate(char, other_char),
                epsilon = 1e-10
            );
        }
        assert_relative_eq!(pip_model.rate(char, GAP), pip_model.mu());
        assert_relative_eq!(pip_model.rate(GAP, char), 0.0);
    }
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<JC69>::new(&[], &[0.1, 0.4]);
    assert_eq!(pip_jc69.lambda(), 0.1);
    assert_eq!(pip_jc69.mu(), 0.4);
    assert_eq!(
        pip_jc69.freqs(),
        &frequencies!(&[0.25, 0.25, 0.25, 0.25, 0.0])
    );
    assert!(pip_jc69
        .q
        .diagonal()
        .iter()
        .take(pip_jc69.q.nrows() - 2)
        .all(|&x| x == -1.0 - 0.4));
}

#[test]
fn pip_dna_k80_correct() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let pip_k80 = PIPModel::<K80>::new(&[], &[lambda, mu, kappa]);
    assert_eq!(pip_k80.lambda(), lambda);
    assert_eq!(pip_k80.mu(), mu);
    assert_eq!(
        pip_k80.freqs(),
        &frequencies!(&[0.25, 0.25, 0.25, 0.25, 0.0])
    );
    assert_eq!(pip_k80.subst_q.params(), &[kappa]);
    assert!(pip_k80
        .q
        .diagonal()
        .iter()
        .take(pip_k80.q.nrows() - 2)
        .all(|&x| x == -1.0 - mu));
}

#[test]
fn pip_dna_hky_correct() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let freqs = &[0.22, 0.26, 0.33, 0.19];
    let pip_hky = PIPModel::<HKY>::new(freqs, &[lambda, mu, kappa]);
    assert_eq!(pip_hky.lambda(), lambda);
    assert_eq!(pip_hky.mu(), mu);
    assert_eq!(pip_hky.freqs(), &frequencies!(freqs).insert_row(4, 0.0));
    assert_eq!(pip_hky.subst_q.params(), &[kappa]);
}

#[test]
fn pip_dna_hky_as_k80() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let pip_k80 = PIPModel::<K80>::new(&[], &[lambda, mu, kappa]);
    let pip_hky = PIPModel::<HKY>::new(&[0.25; 4], &[lambda, mu, kappa]);
    assert_eq!(pip_k80.lambda(), pip_hky.lambda());
    assert_eq!(pip_k80.mu(), pip_hky.mu());
    assert_eq!(pip_k80.freqs(), pip_hky.freqs());
    assert_eq!(pip_k80.subst_q.params(), pip_hky.subst_q.params());
    assert!(pip_hky
        .q
        .diagonal()
        .iter()
        .take(pip_hky.q.nrows() - 2)
        .all(|&x| x == -1.0 - mu));
}

#[cfg(test)]
fn pip_too_few_params_template<Q: QMatrix + QMatrixMaker>(
    freqs: &[f64],
    params: &[f64],
    expected: &[f64],
) {
    let model = PIPModel::<Q>::new(freqs, params);
    assert_eq!(model.params().len(), 2);
    assert_eq!(model.params(), expected);
}

#[test]
fn pip_dna_too_few_params() {
    // Not providing DNA model parameters makes no difference as defaults are taken
    pip_too_few_params_template::<JC69>(&[], &[0.2], &[0.2, 1.5]);
    pip_too_few_params_template::<K80>(&[], &[], &[1.5, 1.5]);
    pip_too_few_params_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.6], &[0.6, 1.5]);
}

#[test]
fn pip_protein_too_few_params() {
    // Not providing substitution model parameters makes no difference as defaults are taken
    pip_too_few_params_template::<WAG>(&[], &[0.2], &[0.2, 1.5]);
    pip_too_few_params_template::<HIVB>(&[], &[1.5], &[1.5, 1.5]);
    pip_too_few_params_template::<BLOSUM>(&[0.22, 0.26, 0.33, 0.19], &[], &[1.5, 1.5]);
}

#[test]
fn pip_dna_subst_rates() {
    compare_pip_subst_rates_template::<JC69>(nucls);
    compare_pip_subst_rates_template::<K80>(nucls);
    compare_pip_subst_rates_template::<HKY>(nucls);
    compare_pip_subst_rates_template::<TN93>(nucls);
    compare_pip_subst_rates_template::<GTR>(nucls);
}

#[test]
fn pip_protein_subst_rates() {
    compare_pip_subst_rates_template::<WAG>(aas);
    compare_pip_subst_rates_template::<HIVB>(aas);
    compare_pip_subst_rates_template::<BLOSUM>(aas);
}

#[cfg(test)]
fn pip_normalised_check_template<Q: QMatrix + QMatrixMaker>(
    chars: &[u8],
    freqs: &[f64],
    params: &[f64],
) {
    let pip = PIPModel::<Q>::new(freqs, params);
    assert_eq!(pip.lambda(), params[0]);
    assert_eq!(pip.mu(), params[1]);
    let stat_dist = frequencies!(freqs).insert_row(freqs.len(), 0.0);
    assert_relative_eq!(pip.freqs(), &stat_dist);
    assert_relative_eq!(pip.q.sum(), 0.0, epsilon = 1e-10);
    for &char in chars {
        let mut sum = 0.0;
        for &other_char in chars {
            sum += pip.rate(char, other_char);
        }
        assert_relative_eq!(sum, 0.0, epsilon = 1e-14);
        assert_relative_eq!(pip.rate(char, GAP), params[1]);
        assert_relative_eq!(pip.rate(GAP, char), 0.0);
    }
}

#[test]
fn pip_dna_normalised() {
    pip_normalised_check_template::<JC69>(nucls, &[0.25; 4], &[0.2, 0.5]);
    pip_normalised_check_template::<K80>(nucls, &[0.25; 4], &[0.1, 1.5]);
    pip_normalised_check_template::<HKY>(nucls, &[0.2, 0.5, 0.1, 0.2], &[0.05, 0.7]);
    pip_normalised_check_template::<TN93>(nucls, &[0.25, 0.45, 0.15, 0.15], &[0.05, 0.7]);
    pip_normalised_check_template::<GTR>(nucls, &[0.6, 0.1, 0.06, 0.24], &[0.05, 0.7]);
}

#[test]
fn pip_protein_normalised() {
    pip_normalised_check_template::<WAG>(aas, &WAG_PI, &[0.2, 0.5]);
    pip_normalised_check_template::<HIVB>(aas, &HIVB_PI, &[0.1, 1.5]);
    pip_normalised_check_template::<BLOSUM>(aas, &BLOSUM_PI, &[0.05, 0.7]);
}

#[cfg(test)]
fn pip_infinity_p_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model = PIPModel::<Q>::new(freqs, params);
    let p_inf = model.p(10000.0);
    assert_eq!(p_inf.shape(), model.q().shape());
    for row in p_inf.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(row.sum() - row[row.len() - 1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(row[row.len() - 1], 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pip_dna_infinity_p() {
    pip_infinity_p_template::<JC69>(&[], &[0.2, 0.5]);
    pip_infinity_p_template::<K80>(&[], &[0.2, 0.5]);
    pip_infinity_p_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.2, 0.3, 0.5]);
    pip_infinity_p_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.1, 0.4, 0.5970915, 0.2940435, 0.00135],
    );
    pip_infinity_p_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[0.2, 0.5, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    );
}

#[test]
fn pip_protein_infinity_p() {
    pip_infinity_p_template::<WAG>(&[], &[0.2, 0.5]);
    pip_infinity_p_template::<HIVB>(&[], &[0.2, 0.5]);
    pip_infinity_p_template::<BLOSUM>(&[], &[0.2, 0.3]);
}

#[test]
fn pip_dna_tn93_correct() {
    let pip_tn93 = PIPModel::<TN93>::new(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.2, 0.5, 0.5970915, 0.2940435, 0.00135],
    );
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    let mut diff = SubstMatrix::zeros(4, 4);
    diff.fill_diagonal(-0.5);
    diff = diff.insert_column(4, 0.5).insert_row(4, 0.0);
    let expected_q = tn93.q().clone().insert_column(4, 0.0).insert_row(4, 0.0) + diff;
    assert_relative_eq!(pip_tn93.q, expected_q, epsilon = 1e-10);
    assert_relative_eq!(
        pip_tn93.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19, 0.0])
    );
}

#[test]
fn pip_p_example_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let epsilon = 1e-3;
    let mut pip_hky = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    pip_hky.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.225, 0.109, 0.173, 0.0996, 0.393, 0.0922, 0.242, 0.173, 0.0996, 0.393, 0.115, 0.136,
            0.276, 0.0792, 0.393, 0.115, 0.136, 0.138, 0.217, 0.393, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.p(2.0), expected_p, epsilon = epsilon);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.437, 0.0859, 0.162, 0.0935, 0.221, 0.0727, 0.45, 0.162, 0.0935, 0.221, 0.108, 0.128,
            0.48, 0.0625, 0.221, 0.108, 0.128, 0.108, 0.434, 0.221, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.p(1.0), expected_p, epsilon = epsilon);
}

#[cfg(test)]
fn setup_example_phylo_info() -> PhyloInfo {
    let tree = tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"-A--"),
            record!("B", b"CA--"),
            record!("C", b"-A-G"),
            record!("D", b"-CAA"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn assert_values<Q: QMatrix>(
    tmp: &PIPModelInfo<Q>,
    idx: usize,
    exp_survins: f64,
    exp_anc: &[f64],
    exp_pnu: &[f64],
) {
    let e = 1e-3;
    assert_relative_eq!(tmp.cache.surv_ins_weights[idx], exp_survins, epsilon = e);
    assert_eq!(tmp.cache.anc[idx].nrows(), exp_pnu.len());
    assert_eq!(tmp.cache.anc[idx].ncols(), 3);
    assert_relative_eq!(tmp.cache.anc[idx].as_slice(), exp_anc,);
    assert_relative_eq!(tmp.cache.pnu[idx].as_slice(), exp_pnu, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_leaf_values() {
    let info = setup_example_phylo_info();
    let tree = info.tree.clone();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let nu = 7.5;
    let ib = 0.1333 * 0.78694; // iota times beta

    let c = PIPB::new(model, info).build().unwrap();
    c.cost();
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("A")),
        nu * ib,
        &[[0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33 * nu * ib, 0.0, 0.0],
    );
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("B")),
        nu * ib,
        &[[1.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.26 * nu * ib, 0.33 * nu * ib, 0.0, 0.0],
    );

    let ib = 0.0667 * 0.8848; // iota times beta
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("C")),
        nu * ib,
        &[[0.0, 1.0, 0.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33 * nu * ib, 0.0, 0.19 * nu * ib],
    );
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("D")),
        nu * ib,
        &[[0.0, 1.0, 1.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.26 * nu * ib, 0.33 * nu * ib, 0.33 * nu * ib],
    );
}

#[test]
fn pip_hky_likelihood_example_internals() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    let ib_e = 0.1333 * 0.78694; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("E")),
        nu * ib_e,
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[nu * ib_e * (0.0619 + 0.26), nu * ib_e * (0.0431), 0.0, 0.0],
    );

    let ib_f = 0.2 * 0.70351; // iota times beta
    let ib_d = 0.0667 * 0.8848; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("F")),
        nu * ib_f,
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[
            0.0,
            nu * (0.0449 * ib_f),
            nu * (0.0567 * ib_f + 0.33 * ib_d),
            nu * (0.0261 * ib_f),
        ],
    );

    let ib_r = 0.2667 * 1.0; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("R")),
        nu * ib_r,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        &[
            nu * (0.0207 * ib_r + (0.0619 + 0.26) * ib_e),
            nu * (0.000557 * ib_r),
            nu * (0.013 * ib_r + 0.0567 * ib_f + 0.33 * ib_d),
            nu * (0.00598 * ib_r + 0.0261 * ib_f),
        ],
    );
}

#[cfg(test)]
fn assert_c0_values<Q: QMatrix>(tmp: &PIPModelInfo<Q>, idx: usize, exp_f1: f64, exp_pnu: f64) {
    let e = 1e-3;
    assert_relative_eq!(tmp.cache.c0_f1[idx], exp_f1, epsilon = e);
    assert_relative_eq!(tmp.cache.c0_pnu[idx], exp_pnu, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_c0() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("A")),
        -1.0,
        nu * 0.028408 - 1.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("B")),
        -1.0,
        nu * 0.028408 - 1.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("C")),
        -1.0,
        nu * 0.00768 - 0.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("D")),
        -1.0,
        nu * 0.00768 - 0.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("E")),
        -0.84518,
        nu * (0.044652 + 0.028408 * 2.0) - 3.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("F")),
        -0.9512,
        nu * 0.066164 + nu * 0.00768 * 2.0 - 2.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("R")),
        -0.732,
        nu * (0.071467 + 0.044652 + 0.028408 * 2.0 + 0.066164 + 0.00768 * 2.0 - 1.0),
    );
}

#[test]
fn pip_hky_likelihood_example_final() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    assert_relative_eq!(
        tmp.cache.pnu[0].as_slice(),
        &[
            nu * 0.0392204949,
            nu * 0.000148719,
            nu * 0.03102171,
            nu * 0.00527154
        ]
        .as_slice(),
        epsilon = 1e-3
    );
    assert_relative_eq!(tmp.cache.c0_pnu[0], -5.591, epsilon = 1e-3);
    drop(tmp);
    assert_relative_eq!(
        c.cost(),
        -20.769363665853653 - 0.709020450847471,
        epsilon = 1e-3
    );
    assert_relative_eq!(c.cost(), -21.476307347643274, epsilon = 1e-2); // value from the python script
}

#[cfg(test)]
fn setup_example_phylo_info_2() -> PhyloInfo {
    let tree = tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"--A--"),
            record!("B", b"-CA--"),
            record!("C", b"--A-G"),
            record!("D", b"T-CAA"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn pip_hky_likelihood_example_2() {
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info_2();
    let c = PIPB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), -24.9549393298, epsilon = 1e-2);
}

#[test]
fn pip_likelihood_huelsenbeck_example() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]);
    let mut c = PIPB::new(model, info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -372.1419415285655, epsilon = 1e-4);

    // Check that model update works
    c.set_param(0, 1.2);
    c.set_param(1, 0.45);
    c.set_freqs(frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    c.set_param(2, 1.0);

    assert_relative_eq!(c.cost(), -361.1613531649497, epsilon = 1e-1); // value from the python script

    let model = PIPModel::<GTR>::new(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5, 0.25, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031],
    );
    let c = PIPB::new(model, info).build().unwrap();

    assert_relative_eq!(c.cost(), -359.2343309917135, epsilon = 1e-4);
}

#[test]
fn pip_likelihood_huelsenbeck_example_model_comp() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let jc69 = PIPModel::<JC69>::new(&[], &[1.1, 0.55]);
    let c = PIPB::new(jc69, info.clone()).build().unwrap();

    let hky_as_jc = PIPModel::<HKY>::new(&[0.25, 0.25, 0.25, 0.25], &[1.1, 0.55, 1.0]);
    let c_hky = PIPB::new(hky_as_jc, info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), c_hky.cost());
}

#[test]
fn pip_likelihood_huelsenbeck_example_reroot() {
    let phylo = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let model_gtr = PIPModel::<GTR>::new(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5, 0.25, 1.25453, 1.07461, 1.0, 1.14689, 1.53244, 1.47031],
    );
    let phylo_rerooted = PIB::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();
    let c = PIPB::new(model_gtr.clone(), phylo).build().unwrap();
    let c_rerooted = PIPB::new(model_gtr, phylo_rerooted).build().unwrap();

    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = 1e-4);
    assert_relative_eq!(c.cost(), -359.2343309917135, epsilon = 1e-4);
}

#[test]
fn pip_likelihood_protein_example() {
    let info = PIB::with_attrs(
        PathBuf::from("./data/phyml_protein_example/seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/example_tree.newick"),
    )
    .build()
    .unwrap();

    let model_wag = PIPModel::<WAG>::new(&WAG_PI, &[0.5, 0.25]);
    let c = PIPB::new(model_wag, info.clone()).build().unwrap();
    assert!(c.cost() <= 0.0);
    // The cost is the same when initialising the model with default frequencies
    assert_eq!(
        PIPB::new(PIPModel::<WAG>::new(&[], &[0.5, 0.25]), info.clone())
            .build()
            .unwrap()
            .cost(),
        c.cost(),
    );

    // The cost is different when initialising the model with equal frequencies
    assert_ne!(
        PIPB::new(
            PIPModel::<WAG>::new(&[0.05; 20], &[0.5, 0.25]),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );

    // The cost is different when initialising the model with default frequencies but different mu/lambda
    assert_ne!(
        PIPB::new(PIPModel::<WAG>::new(&[], &[0.5, 0.2]), info.clone())
            .build()
            .unwrap()
            .cost(),
        c.cost(),
    );
    assert_ne!(
        PIPB::new(PIPModel::<WAG>::new(&[], &[0.1, 0.25]), info.clone())
            .build()
            .unwrap()
            .cost(),
        c.cost(),
    );
    assert_ne!(
        PIPB::new(PIPModel::<WAG>::new(&[], &[0.1, 0.2]), info.clone())
            .build()
            .unwrap()
            .cost(),
        c.cost(),
    );
}

#[test]
fn designation() {
    let model = PIPModel::<JC69>::new(&[], &[]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("lambda = 1.5"));
    assert!(format!("{}", model).contains("mu = 1.5"));
    assert!(format!("{}", model).contains("JC69"));

    let model = PIPModel::<JC69>::new(&[], &[2.0, 1.0]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("lambda = 2.0"));
    assert!(format!("{}", model).contains("mu = 1.0"));
    assert!(format!("{}", model).contains("JC69"));

    let model = PIPModel::<K80>::new(&[], &[2.0, 5.0, 1.3]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("lambda = 2.0"));
    assert!(format!("{}", model).contains("mu = 5.0"));
    assert!(format!("{}", model).contains("K80"));
    assert!(format!("{}", model).contains("kappa = 1.3"));

    let model = PIPModel::<HKY>::new(&[], &[2.0, 5.0, 2.5]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("lambda = 2.0"));
    assert!(format!("{}", model).contains("mu = 5.0"));
    assert!(format!("{}", model).contains("HKY"));
    assert!(format!("{}", model).contains("kappa = 2.5"));

    let model = PIPModel::<TN93>::new(&[], &[2.5, 0.3, 0.1]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("lambda = 2.5"));
    assert!(format!("{}", model).contains("mu = 0.3"));
    assert!(format!("{}", model).contains("TN93"));
    assert!(format!("{}", model).contains("0.1"));
    assert!(format!("{}", model).contains("1.0"));

    let model = PIPModel::<GTR>::new(&[], &[1.4, 1.7]);
    assert!(format!("{}", model).contains("GTR"));

    let model = PIPModel::<WAG>::new(&[], &[2.0, 1.0]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("WAG"));

    let model = PIPModel::<HIVB>::new(&[], &[2.0, 1.0]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("HIVB"));

    let model = PIPModel::<BLOSUM>::new(&[], &[2.0, 1.0]);
    assert!(format!("{}", model).contains("PIP"));
    assert!(format!("{}", model).contains("BLOSUM"));
}

#[test]
fn pip_logl_correct_w_diff_info() {
    let tree1 = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let tree2 = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let seqs = Sequences::new(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);

    let info1 = PhyloInfo {
        msa: Alignment::from_aligned(seqs.clone(), &tree1).unwrap(),
        tree: tree1,
    };

    let info2 = PhyloInfo {
        msa: Alignment::from_aligned(seqs, &tree2).unwrap(),
        tree: tree2,
    };

    let pip_wag = PIPModel::<WAG>::new(&[], &[50.0, 0.1]);

    let c1 = PIPB::new(pip_wag.clone(), info1).build().unwrap();
    let c2 = PIPB::new(pip_wag, info2).build().unwrap();

    assert_relative_eq!(c1.cost(), -854.2260753055998, epsilon = 1e-2);
    assert_relative_eq!(c2.cost(), -1125.1290016747846, epsilon = 1e-2);
    assert_ne!(c1.cost(), c2.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn hiv_subset_valid_pip_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PIB::new(alignment).build().unwrap();
    let pip = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let c = PIPB::new(pip, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[cfg(test)]
fn avg_rate_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let mu = params[1];
    let model = PIPModel::<Q>::new(freqs, params);
    let n = model.q().nrows();
    let avg_rate = model
        .q()
        .view((0, 0), (n - 1, n - 1))
        .diagonal()
        .component_mul(&model.freqs().view((0, 0), (n - 1, 1)))
        .sum();
    assert_relative_eq!(avg_rate, -1.0 - mu, epsilon = 1e-6);
}

#[test]
fn dna_avg_rate() {
    avg_rate_template::<JC69>(&[], &[0.24, 1.4]);
    avg_rate_template::<K80>(&[], &[0.4, 4.4]);
    avg_rate_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5, 1.5, 0.5]);
    avg_rate_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[1.5, 0.25, 0.5, 0.2, 0.001]);
    avg_rate_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn protein_avg_rate() {
    avg_rate_template::<WAG>(&[], &[0.5, 1.0]);
    avg_rate_template::<HIVB>(&[], &[5.0, 1.5]);
    avg_rate_template::<BLOSUM>(&[], &[4.5, 1.3]);
    let freqs = &[1.0 / 20.0; 20];
    avg_rate_template::<WAG>(freqs, &[0.5, 1.0]);
    avg_rate_template::<HIVB>(freqs, &[2.5, 0.03]);
    avg_rate_template::<BLOSUM>(freqs, &[0.25, 1.5]);
}

#[test]
fn logl_not_inf_for_empty_col() {
    let tree = tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;");
    let msa = Alignment::from_aligned(
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_empty_col.fasta")).unwrap()),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };
    let model = PIPModel::<WAG>::new(&[], &[0.5, 0.5]);
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn blen_leading_to_small_probs() {
    let fldr = Path::new("./data/");
    let seq_file = fldr.join("p105.msa.fa");
    let tree_file = fldr.join("p105.newick");
    let info = PIB::with_attrs(seq_file, tree_file).build().unwrap();

    let model = PIPModel::<WAG>::new(&[], &[]);
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn blen_leading_to_minusinf() {
    let tree = tree!("((284811:0.0000000000000002,(284593:0.1,(237561:0.3,(284812:0.3,(284813:400.9,284591:0.2):40000000000000.2):0.05):0.1):0.04):0);");
    let msa = Alignment::from_aligned(
        Sequences::with_alphabet(
            vec![
                record!("284813", b"-"),
                record!("284811", b"W"),
                record!("284593", b"W"),
                record!("237561", b"W"),
                record!("284591", b"W"),
                record!("284812", b"W"),
            ],
            protein_alphabet(),
        ),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };

    let model = PIPModel::<WAG>::new(&[], &[]);
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl.is_sign_negative());
}
