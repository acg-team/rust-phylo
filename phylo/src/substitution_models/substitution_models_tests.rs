use rstest::*;

use std::collections::HashMap;
use std::iter::repeat;
use std::ops::Mul;

use approx::assert_relative_eq;
use nalgebra::dvector;
use rand::Rng;

use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvolutionaryModel,
    ProteinModelType::{self, *},
};
use crate::sequences::AMINOACIDS;
use crate::substitution_models::dna_models::dna_model_generics::{
    gtr_params, hky_params, jc69_params, k80_params, tn93_params,
};
use crate::substitution_models::dna_models::{DNASubstModel, DNA_SETS};
use crate::substitution_models::protein_models::{
    ProteinSubstArray, ProteinSubstModel, BLOSUM_PI_ARR, HIVB_PI_ARR, PROTEIN_SETS, WAG_PI_ARR,
};
use crate::substitution_models::{FreqVector, ParsimonyModel, SubstMatrix};
use crate::Rounding as R;

#[cfg(test)]

fn check_pi_convergence(substmat: SubstMatrix, pi: &FreqVector, epsilon: f64) {
    assert_eq!(substmat.row(0).len(), pi.len());
    for row in substmat.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(row, pi.transpose().as_view(), epsilon = epsilon);
    }
}

#[cfg(test)]
pub(crate) fn gtr_char_probs_data() -> (Vec<f64>, HashMap<u8, FreqVector>) {
    (
        [0.21, 0.30, 0.34, 0.15]
            .into_iter()
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
        HashMap::from([
            (b'T', dvector![1.0, 0.0, 0.0, 0.0]),
            (b'C', dvector![0.0, 1.0, 0.0, 0.0]),
            (b'A', dvector![0.0, 0.0, 1.0, 0.0]),
            (b'G', dvector![0.0, 0.0, 0.0, 1.0]),
            (b'X', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'N', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'Z', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'P', dvector![0.21, 0.30, 0.34, 0.15]),
            (b'V', dvector![0.0, 0.37974684, 0.43037975, 0.18987342]),
            (b'D', dvector![0.3, 0.0, 0.48571429, 0.21428571]),
            (b'B', dvector![0.31818182, 0.45454545, 0.0, 0.22727273]),
            (b'H', dvector![0.24705882, 0.35294118, 0.4, 0.0]),
            (b'M', dvector![0.0, 0.46875, 0.53125, 0.0]),
            (b'R', dvector![0.0, 0.0, 0.69387755, 0.30612245]),
            (b'W', dvector![0.38181818, 0.0, 0.61818182, 0.0]),
            (b'S', dvector![0.0, 0.66666667, 0.0, 0.33333333]),
            (b'Y', dvector![0.41176471, 0.58823529, 0.0, 0.0]),
            (b'K', dvector![0.58333333, 0.0, 0.0, 0.41666667]),
        ]),
    )
}

#[cfg(test)]
pub(crate) fn protein_char_probs_data(pi: &[f64]) -> HashMap<u8, FreqVector> {
    HashMap::from([
        (b'A', compile_aa_probability(&[b'A'], pi)),
        (b'R', compile_aa_probability(&[b'R'], pi)),
        (b'W', compile_aa_probability(&[b'W'], pi)),
        (b'B', compile_aa_probability(&[b'D', b'N'], pi)),
        (b'Z', compile_aa_probability(&[b'E', b'Q'], pi)),
        (b'J', compile_aa_probability(&[b'I', b'L'], pi)),
        (b'X', frequencies!(pi)),
    ])
}

#[cfg(test)]
fn compile_aa_probability(chars: &[u8], pi: &[f64]) -> FreqVector {
    let mut char_probs = frequencies!(&[0.0; 20]);
    if chars.len() == 1 {
        char_probs[AMINOACIDS.iter().position(|&x| x == chars[0]).unwrap()] = 1.0;
        char_probs
    } else {
        for c in chars {
            let position = AMINOACIDS.iter().position(|&x| x == *c).unwrap();
            char_probs[position] = pi[position];
        }
        char_probs.scale_mut(1.0 / char_probs.sum());
        char_probs
    }
}

#[rstest]
#[case::jc69("jc69", JC69, &["jc70"])]
#[case::k80("k80", K80, &["K 80", "k89"])]
#[case::hky("hky", HKY, &["hkz", "hky3"])]
#[case::tn93("tn93", TN93, &["TN92", "tn993"])]
#[case::gtr("gtr", GTR, &["ctr", "GTP", "gtr1"])]
fn dna_type_by_name(
    #[case] name: &str,
    #[case] model_type: DNAModelType,
    #[case] wrong_name: &[&str],
) {
    assert_eq!(DNAModelType::get_model_type(name), model_type);
    assert_eq!(
        DNAModelType::get_model_type(&name.to_ascii_uppercase()),
        model_type
    );
    for _ in 0..10 {
        assert_eq!(
            DNAModelType::get_model_type(&random_capitalise(name)),
            model_type
        );
    }
    for name in wrong_name {
        assert_eq!(DNAModelType::get_model_type(name), GTR);
    }
}

#[rstest]
#[case::wag("wag", WAG, &["weg", "wag1"])]
#[case::blosum("blosum", BLOSUM, &["BLOS", "blosum1", "blosum62"])]
#[case::hivb("hivb", HIVB, &["Hiv1", "HIB", "HIVA"])]
fn protein_type_by_name(
    #[case] name: &str,
    #[case] model_type: ProteinModelType,
    #[case] wrong_name: &[&str],
) {
    assert_eq!(ProteinModelType::get_model_type(name), model_type);
    assert_eq!(
        ProteinModelType::get_model_type(&name.to_ascii_uppercase()),
        model_type
    );
    for _ in 0..10 {
        assert_eq!(
            ProteinModelType::get_model_type(&random_capitalise(name)),
            model_type
        );
    }
    for name in wrong_name {
        assert_eq!(ProteinModelType::get_model_type(name), WAG);
    }
}

fn random_capitalise(input: &str) -> String {
    let mut rng = rand::thread_rng();
    input
        .chars()
        .map(|c| {
            if c.is_alphabetic() && rng.gen_bool(0.5) {
                c.to_uppercase().collect::<String>()
            } else {
                c.to_string()
            }
        })
        .collect()
}

#[test]
fn dna_type_by_name_given_protein() {
    assert_eq!(DNAModelType::get_model_type("wag"), GTR);
    assert_eq!(DNAModelType::get_model_type("BLOSUM"), GTR);
    assert_eq!(DNAModelType::get_model_type("HIv"), GTR);
}

#[test]
fn protein_type_by_name_given_dna() {
    assert_eq!(ProteinModelType::get_model_type("k80"), WAG);
    assert_eq!(ProteinModelType::get_model_type("gtr"), WAG);
    assert_eq!(ProteinModelType::get_model_type("TN93"), WAG);
}

#[test]
fn dna_jc69_correct() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let jc69_2 = DNASubstModel::new(JC69, &[1.0, 2.0]).unwrap();
    assert_eq!(jc69, jc69_2);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'A', b'A'), -1.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'A', b'C'), 1.0 / 3.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&jc69, b'G', b'T'), 1.0 / 3.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&jc69),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
    let jc69_3 = DNASubstModel::new(JC69, &[4.0]).unwrap();
    assert_eq!(jc69.q, jc69_3.q);
    assert_eq!(jc69.pi, jc69_3.pi);
}

#[test]
fn dna_j69_param_print() {
    let params = jc69_params(&[0.1, 0.4, 0.75, 1.5]).unwrap();
    assert_relative_eq!(params.pi, dvector![0.25, 0.25, 0.25, 0.25]);
    assert_eq!(format!("{}", params), format!("[lambda = {}]", 1.0));
}

#[test]
fn dna_k80_correct() {
    let k80 = DNASubstModel::new(K80, &[]).unwrap();
    let k801 = DNASubstModel::new(K80, &[2.0]).unwrap();
    let k802 = DNASubstModel::new(K80, &[2.0, 1.0]).unwrap();
    let k803 = DNASubstModel::new(K80, &[2.0, 1.0, 3.0, 6.0]).unwrap();
    assert_eq!(k80, k801);
    assert_eq!(k80, k802);
    assert_eq!(k80, k803);
    assert_eq!(k802, k803);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'A', b'A'), -1.0);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'T', b'A'), 1.0 * 0.25);
    assert_relative_eq!(EvolutionaryModel::get_rate(&k80, b'A', b'G'), 2.0 * 0.25);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&k80),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn dna_k80_params() {
    let params = k80_params(&[0.75, 1.5]).unwrap();
    assert_relative_eq!(params.pi, dvector![0.25, 0.25, 0.25, 0.25]);
    assert_eq!(
        format!("{}", params),
        format!("[alpha = {}, beta = {}]", 0.75, 1.5)
    );
}

#[test]
fn dna_hky_incorrect() {
    let hky = DNASubstModel::new(HKY, &[2.0, 1.0, 3.0, 6.0]);
    assert!(hky.is_err());
    let hky = DNASubstModel::new(HKY, &[2.0, 1.0, 3.0, 6.0, 0.5]);
    assert!(hky.is_err());
    let hky = DNASubstModel::new(HKY, &[2.0, 1.0, 3.0, 6.0, 0.5, 1.0]);
    assert!(hky.is_err());
}

#[test]
fn dna_hky_correct() {
    let hky = DNASubstModel::new(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5]).unwrap();
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&hky),
        &dvector![0.22, 0.26, 0.33, 0.19]
    );
    let hky2 = DNASubstModel::new(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5, 1.0]).unwrap();
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&hky2),
        &dvector![0.22, 0.26, 0.33, 0.19]
    );
    assert_eq!(hky, hky2);
    let hky3 = DNASubstModel::new(HKY, &[0.22, 0.26, 0.33, 0.19]).unwrap();
    let hky4 = DNASubstModel::new(HKY, &[0.22, 0.26, 0.33, 0.19, 2.0, 1.0]).unwrap();
    assert_relative_eq!(
        hky3.q
            .diagonal()
            .component_mul(&dvector![0.22, 0.26, 0.33, 0.19])
            .sum(),
        -1.0
    );
    assert_eq!(hky3, hky4);
}

#[test]
fn dna_hky_params() {
    let params = hky_params(&[0.22, 0.26, 0.33, 0.19, 0.75, 1.5]).unwrap();
    assert_relative_eq!(params.pi, dvector![0.22, 0.26, 0.33, 0.19]);
    assert_eq!(
        format!("{}", params),
        format!(
            "[pi = [{}, {}, {}, {}], alpha = {}, beta = {}]",
            0.22, 0.26, 0.33, 0.19, 0.75, 1.5
        )
    );
}

#[test]
fn dna_gtr_correct() {
    let gtr = DNASubstModel::new(
        GTR,
        &repeat(0.25)
            .take(4)
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    assert_eq!(gtr.pi, dvector![0.25, 0.25, 0.25, 0.25]);
    assert_eq!(gtr.q[(0, 0)], -1.0);
    let gtr2 = DNASubstModel::new(
        GTR,
        &repeat(0.25)
            .take(4)
            .chain(repeat(1.0).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    assert_relative_eq!(gtr.q, gtr2.q);
    assert!(EvolutionaryModel::get_rate(&gtr, b'T', b'T') < 0.0);
    assert!(EvolutionaryModel::get_rate(&gtr, b'A', b'A') < 0.0);
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&gtr, b'T', b'C'),
        EvolutionaryModel::get_rate(&gtr, b'C', b'T')
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&gtr, b'A', b'G'),
        EvolutionaryModel::get_rate(&gtr, b'G', b'A')
    );
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&gtr),
        &dvector![0.25, 0.25, 0.25, 0.25]
    );
}

#[test]
fn dna_gtr_incorrect() {
    let gtr = DNASubstModel::new(GTR, &[2.0, 1.0, 3.0, 6.0]);
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new(GTR, &[0.22, 0.26, 0.33, 0.19, 0.5, 0.6, 0.7]);
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new(
        GTR,
        &repeat(0.3)
            .take(4)
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
    );
    assert!(gtr.is_err());
    let gtr = DNASubstModel::new(
        GTR,
        &repeat(0.25)
            .take(4)
            .chain(repeat(0.7).take(7))
            .collect::<Vec<f64>>(),
    );
    assert!(gtr.is_err());
}

#[test]
fn dna_gtr_params() {
    let params = gtr_params(&[0.22, 0.26, 0.33, 0.19, 0.75, 1.5, 3.0, 1.25, 0.45, 0.1]).unwrap();
    assert_relative_eq!(params.pi, dvector![0.22, 0.26, 0.33, 0.19]);
    assert_eq!(
        format!("{}", params),
        format!(
            "[pi = [{}, {}, {}, {}], rtc = {}, rta = {}, rtg = {}, rca = {}, rcg = {}, rag = {}]",
            0.22, 0.26, 0.33, 0.19, 0.75, 1.5, 3.0, 1.25, 0.45, 0.1
        )
    );
}

#[test]
fn dna_tn93_correct() {
    let tn93 = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    let expected_pi = dvector![0.22, 0.26, 0.33, 0.19];
    let expected_q = SubstMatrix::from_column_slice(
        4,
        4,
        &[
            -1.4732124694954951,
            1.2409529074850258,
            0.002805744890196536,
            0.002805744890196536,
            1.4665807088459397,
            -1.2475846681345815,
            0.0033158803247777245,
            0.0033158803247777245,
            0.0042086173352948045,
            0.0042086173352948045,
            -0.5339064704940854,
            0.9166789418005613,
            0.002423143314260645,
            0.002423143314260645,
            0.5277848452791111,
            -0.9228005670155355,
        ],
    );
    assert_relative_eq!(tn93.q, expected_q);
    assert_relative_eq!(tn93.q.diagonal().component_mul(&expected_pi).sum(), -1.0);
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&tn93),
        &expected_pi
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&tn93, b'T', b'C') * expected_pi[0],
        EvolutionaryModel::get_rate(&tn93, b'C', b'T') * expected_pi[1],
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&tn93, b'A', b'G') * expected_pi[2],
        EvolutionaryModel::get_rate(&tn93, b'G', b'A') * expected_pi[3],
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&tn93, b'T', b'A') * expected_pi[0],
        EvolutionaryModel::get_rate(&tn93, b'A', b'T') * expected_pi[2],
    );
    assert_relative_eq!(
        EvolutionaryModel::get_rate(&tn93, b'C', b'G') * expected_pi[1],
        EvolutionaryModel::get_rate(&tn93, b'G', b'C') * expected_pi[3],
    );
}

#[test]
fn dna_tn93_incorrect() {
    let tn93 = DNASubstModel::new(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435]);
    assert!(tn93.is_err());
    let tn93 = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 1.19, 0.5970915, 0.2940435, 0.00135],
    );
    assert!(tn93.is_err());
    let tn93 = DNASubstModel::new(
        TN93,
        &[
            0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135, 0.00135,
        ],
    );
    assert!(tn93.is_err());
}

#[test]
fn dna_tn93_params() {
    let params = tn93_params(&[0.22, 0.26, 0.33, 0.19, 0.75, 1.5, 3.0]).unwrap();
    assert_relative_eq!(params.pi, dvector![0.22, 0.26, 0.33, 0.19]);
    assert_eq!(
        format!("{}", params),
        format!(
            "[pi = [{}, {}, {}, {}], alpha1 = {}, alpha2 = {}, beta = {}]",
            0.22, 0.26, 0.33, 0.19, 0.75, 1.5, 3.0
        )
    );
}

#[test]
fn dna_incorrect_gtr_params() {
    assert!(DNASubstModel::new(GTR, &repeat(0.25).take(7).collect::<Vec<f64>>()).is_err());
    assert!(DNASubstModel::new(GTR, &repeat(0.25).take(11).collect::<Vec<f64>>()).is_err());
    assert!(DNASubstModel::new(GTR, &repeat(0.4).take(10).collect::<Vec<f64>>()).is_err());
}

// #[test]
// fn dna_given_protein_type() {
//     assert!(DNASubstModel::new(WAG, &[]).is_err());
//     assert!(DNASubstModel::new(BLOSUM, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435]).is_err());
//     assert!(DNASubstModel::new(HIVB, &[0.26, 0.4]).is_err());
// }

#[test]
fn dna_p_matrix() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let p_inf = EvolutionaryModel::get_p(&jc69, 200000.0);
    assert_eq!(p_inf.nrows(), 4);
    assert_eq!(p_inf.ncols(), 4);
    check_pi_convergence(p_inf, &jc69.pi, 1e-5);
}

#[test]
fn dna_normalisation() {
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    assert_eq!((jc69.q.diagonal().transpose().mul(jc69.pi))[(0, 0)], -1.0);
    let k80 = DNASubstModel::new(K80, &[3.0, 1.5]).unwrap();
    assert_eq!((k80.q.diagonal().transpose().mul(k80.pi))[(0, 0)], -1.0);
    let gtr = DNASubstModel::new(
        GTR,
        &[0.22, 0.26, 0.33, 0.19]
            .into_iter()
            .chain(repeat(0.7).take(6))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    assert_eq!((gtr.q.diagonal().transpose().mul(gtr.pi))[(0, 0)], -1.0);
    let tn93 = DNASubstModel::new(
        TN93,
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
    )
    .unwrap();
    assert_relative_eq!((tn93.q.diagonal().transpose().mul(tn93.pi))[(0, 0)], -1.0);
}

#[test]
fn dna_char_probabilities() {
    let (params, char_probs) = gtr_char_probs_data();
    let gtr = DNASubstModel::new(GTR, &params).unwrap();
    for (&char, expected) in char_probs.iter() {
        let actual = gtr.get_char_probability(&DNA_SETS[char as usize]);
        assert_relative_eq!(actual.sum(), 1.0);
        assert_relative_eq!(actual, expected, epsilon = 1e-4);
    }
}

#[rstest]
#[case::wag(WAG, &WAG_PI_ARR, 1e-8)]
#[case::blosum(BLOSUM, &BLOSUM_PI_ARR, 1e-5)]
#[case::hivb(HIVB, &HIVB_PI_ARR, 1e-8)]
fn protein_char_probabilities(
    #[case] model_type: ProteinModelType,
    #[case] pi_array: &[f64],
    #[case] epsilon: f64,
) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let expected = protein_char_probs_data(pi_array);
    for (char, expected_probs) in expected.into_iter() {
        let actual = model.get_char_probability(&PROTEIN_SETS[char as usize]);
        assert_relative_eq!(actual.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(actual, expected_probs, epsilon = epsilon);
    }
}

#[rstest]
#[case::wag(WAG)]
#[case::blosum(BLOSUM)]
#[case::hivb(HIVB)]
fn protein_weird_char_probabilities(#[case] model_type: ProteinModelType) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    assert_eq!(
        EvolutionaryModel::get_char_probability(&model, &PROTEIN_SETS[b'.' as usize]),
        EvolutionaryModel::get_char_probability(&model, &PROTEIN_SETS[b'X' as usize])
    );
}

#[rstest]
#[case::jc69(JC69, &[])]
#[case::k80(K80, &[])]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5])]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135])]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0])]
fn dna_weird_char_probabilities(#[case] model_type: DNAModelType, #[case] params: &[f64]) {
    let model = DNASubstModel::new(model_type, params).unwrap();
    assert_eq!(
        EvolutionaryModel::get_char_probability(&model, &DNA_SETS[b'.' as usize]),
        EvolutionaryModel::get_char_probability(&model, &DNA_SETS[b'X' as usize]),
    );
}

#[rstest]
#[case::wag(WAG, 1e-4)]
#[case::blosum(BLOSUM, 1e-3)]
#[case::hivb(HIVB, 1e-3)]
fn protein_model_correct(#[case] model_type: ProteinModelType, #[case] epsilon: f64) {
    let model_1 = ProteinSubstModel::new(model_type, &[]).unwrap();
    let model_2 = ProteinSubstModel::new(model_type, &[]).unwrap();
    assert_relative_eq!(model_1.q, model_2.q);
    for _ in 0..10 {
        let mut rng = rand::thread_rng();
        let query1 = AMINOACIDS[rng.gen_range(0..AMINOACIDS.len())];
        let query2 = AMINOACIDS[rng.gen_range(0..AMINOACIDS.len())];
        EvolutionaryModel::get_rate(&model_1, query1, query2);
    }
    assert_relative_eq!(
        EvolutionaryModel::get_stationary_distribution(&model_1).sum(),
        1.0,
        epsilon = epsilon
    );
}

#[rstest]
#[case::wag(WAG)]
#[case::blosum(BLOSUM)]
#[case::hivb(HIVB)]
#[should_panic]
fn protein_model_incorrect_access(#[case] model_type: ProteinModelType) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    EvolutionaryModel::get_rate(&model, b'H', b'J');
    EvolutionaryModel::get_rate(&model, b'-', b'L');
}

#[rstest]
#[case::wag(WAG)]
#[case::blosum(BLOSUM)]
#[case::hivb(HIVB)]
#[should_panic]
fn protein_model_gap(#[case] model_type: ProteinModelType) {
    let wag = ProteinSubstModel::new(model_type, &[]).unwrap();
    EvolutionaryModel::get_rate(&wag, b'-', b'L');
}

#[test]
fn protein_model_def_typo() {
    assert_eq!(ProteinModelType::get_model_type("waq"), WAG);
    assert_eq!(ProteinModelType::get_model_type("HIV"), WAG);
}

#[rstest]
#[case::wag(WAG, 1e-2)]
#[case::blosum(BLOSUM, 1e-3)]
// FIXME: This test fails for HIVB
// #[case::hivb(HIVB, 1e-3)]
fn protein_p_matrix(#[case] model_type: ProteinModelType, #[case] epsilon: f64) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    let p_inf = EvolutionaryModel::get_p(&model, 1000000.0);
    assert_eq!(p_inf.nrows(), 20);
    assert_eq!(p_inf.ncols(), 20);
    check_pi_convergence(p_inf, &model.pi, epsilon);
}

#[rstest]
#[case::wag(WAG, 1e-10)]
#[case::blosum(BLOSUM, 1e-10)]
#[case::hivb(HIVB, 1e-10)]
fn protein_normalisation(#[case] model_type: ProteinModelType, #[case] epsilon: f64) {
    let model = ProteinSubstModel::new(model_type, &[]).unwrap();
    assert_relative_eq!(
        (model.q.diagonal().transpose().mul(model.pi))[(0, 0)],
        -1.0,
        epsilon = epsilon
    );
}

const TRUE_MATRIX: ProteinSubstArray = [
    0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0, 4.0, 9.0, 7.0,
    4.0, 5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0, 6.0, 6.0,
    7.0, 6.0, 5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0, 7.0, 4.0, 4.0,
    9.0, 6.0, 6.0, 5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0, 8.0, 8.0, 6.0, 5.0,
    6.0, 9.0, 7.0, 7.0, 5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0, 6.0, 7.0, 7.0, 6.0, 7.0,
    5.0, 6.0, 7.0, 6.0, 5.0, 5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0, 5.0, 7.0, 5.0, 4.0, 6.0, 8.0,
    5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 6.0, 6.0, 3.0, 9.0, 4.0, 0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0,
    8.0, 6.0, 5.0, 5.0, 8.0, 7.0, 5.0, 4.0, 6.0, 5.0, 5.0, 7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0,
    8.0, 8.0, 7.0, 5.0, 6.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0,
    5.0, 7.0, 6.0, 6.0, 5.0, 6.0, 8.0, 4.0, 7.0, 6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0,
    4.0, 6.0, 5.0, 5.0, 8.0, 6.0, 5.0, 8.0, 6.0, 3.0, 6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0,
    4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0,
    6.0, 6.0, 6.0, 0.0, 6.0, 8.0, 6.0, 5.0, 5.0, 8.0, 8.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0,
    6.0, 7.0, 4.0, 3.0, 5.0, 0.0, 5.0, 7.0, 6.0, 5.0, 7.0, 6.0, 4.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0,
    8.0, 8.0, 6.0, 5.0, 4.0, 7.0, 6.0, 0.0, 7.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 6.0, 7.0, 6.0, 8.0,
    6.0, 6.0, 6.0, 6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 5.0, 4.0, 5.0,
    6.0, 6.0, 5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0, 4.0, 7.0, 6.0, 6.0, 4.0, 6.0, 5.0,
    6.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0, 0.0, 9.0, 7.0, 5.0, 7.0, 5.0,
    8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0, 7.0, 0.0, 5.0, 6.0, 6.0,
    6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0, 6.0, 6.0, 0.0, 6.0,
    4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 5.0, 8.0, 7.0,
    0.0,
];

#[rstest]
#[case::jc69(JC69, &[], &[0.1, 0.3, 0.5, 0.7], &R::four())]
#[case::k80(K80, &[], &[0.01], &R::zero())]
#[case::hky(HKY, &[0.22, 0.26, 0.33, 0.19, 0.5], &[0.1, 0.2, 0.3], &R::none())]
#[case::tn93(TN93, &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135], &[0.1, 0.3, 0.5, 0.7], &R::zero())]
#[case::gtr(GTR, &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0], &[0.2, 0.8], &R::four())]
fn dna_scoring_matrices(
    #[case] model_type: DNAModelType,
    #[case] params: &[f64],
    #[case] times: &[f64],
    #[case] rounding: &R,
) {
    let model = DNASubstModel::new(model_type, params).unwrap();
    let scorings = ParsimonyModel::generate_scorings(&model, times, false, rounding);
    for &time in times {
        let (_, avg_0) = ParsimonyModel::get_scoring_matrix(&model, time, rounding);
        let (_, avg_1) = scorings.get(&ordered_float::OrderedFloat(time)).unwrap();
        assert_relative_eq!(avg_0, avg_1);
    }
}

#[test]
fn protein_scoring_matrices() {
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.1, &R::zero());
    for (row, true_row) in mat.row_iter().zip(true_matrix_01.row_iter()) {
        assert_eq!(row, true_row);
    }
    assert_relative_eq!(avg, 5.7675);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.3, &R::zero());
    assert_relative_eq!(avg, 4.7475);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.5, &R::zero());
    assert_relative_eq!(avg, 4.2825);
    let (_, avg) = ParsimonyModel::get_scoring_matrix(&model, 0.7, &R::zero());
    assert_relative_eq!(avg, 4.0075);
}

#[test]
fn generate_protein_scorings() {
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let scorings =
        ParsimonyModel::generate_scorings(&model, &[0.1, 0.3, 0.5, 0.7], false, &R::zero());
    let true_matrix_01 = SubstMatrix::from_row_slice(20, 20, &TRUE_MATRIX);
    let (mat_01, avg_01) = scorings.get(&ordered_float::OrderedFloat(0.1)).unwrap();
    for (row, true_row) in mat_01.row_iter().zip(true_matrix_01.row_iter()) {
        assert_eq!(row, true_row);
    }
    assert_relative_eq!(*avg_01, 5.7675);
    let (_, avg_03) = scorings.get(&ordered_float::OrderedFloat(0.3)).unwrap();
    assert_relative_eq!(*avg_03, 4.7475);
    let (_, avg_05) = scorings.get(&ordered_float::OrderedFloat(0.5)).unwrap();
    assert_relative_eq!(*avg_05, 4.2825);
    let (_, avg_07) = scorings.get(&ordered_float::OrderedFloat(0.7)).unwrap();
    assert_relative_eq!(*avg_07, 4.0075);
}

#[test]
fn matrix_entry_rounding() {
    let model = DNASubstModel::new(K80, &[1.0, 2.0]).unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
    let model = ProteinSubstModel::new(HIVB, &[]).unwrap();
    let (mat_round, avg_round) = model.get_scoring_matrix_corrected(0.1, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.1, true, &R::none());
    assert_ne!(avg_round, avg);
    assert_ne!(mat_round, mat);
    for &element in mat_round.as_slice() {
        assert_eq!(element.round(), element);
    }
}

#[test]
fn matrix_zero_diagonals() {
    let model = ProteinSubstModel::new(HIVB, &[]).unwrap();
    let (mat_zeros, avg_zeros) = model.get_scoring_matrix_corrected(0.5, true, &R::zero());
    let (mat, avg) = model.get_scoring_matrix_corrected(0.5, false, &R::zero());
    assert_ne!(avg_zeros, avg);
    assert!(avg_zeros < avg);
    assert_ne!(mat_zeros, mat);
    for &element in mat_zeros.diagonal().iter() {
        assert_eq!(element, 0.0);
    }
}
