use rstest::*;

use rand::Rng;

use crate::evolutionary_models::{
    DNAModelType::{self, UNDEF as DNAUNDEF, *},
    ProteinModelType::{self, UNDEF as ProteinUNDEF, *},
};

#[cfg(test)]
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
        assert_eq!(DNAModelType::get_model_type(name), DNAUNDEF);
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
        assert_eq!(ProteinModelType::get_model_type(name), ProteinUNDEF);
    }
}

#[test]
fn dna_type_by_name_given_protein() {
    assert_eq!(DNAModelType::get_model_type("wag"), DNAUNDEF);
    assert_eq!(DNAModelType::get_model_type("BLOSUM"), DNAUNDEF);
    assert_eq!(DNAModelType::get_model_type("HIv"), DNAUNDEF);
}

#[test]
fn protein_type_by_name_given_dna() {
    assert_eq!(ProteinModelType::get_model_type("k80"), ProteinUNDEF);
    assert_eq!(ProteinModelType::get_model_type("gtr"), ProteinUNDEF);
    assert_eq!(ProteinModelType::get_model_type("TN93"), ProteinUNDEF);
    assert_eq!(ProteinModelType::get_model_type("waq"), ProteinUNDEF);
    assert_eq!(ProteinModelType::get_model_type("HIV"), ProteinUNDEF);
}
