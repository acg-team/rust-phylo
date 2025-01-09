use std::path::Path;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::{EvoModel, FrequencyOptimisation::Empirical};
use crate::io::write_newick_to_file;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, ModelOptimiser, TopologyOptimiser};
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, SubstModel, SubstitutionCostBuilder as SCB,
};
use crate::tree::tree_parser::from_newick;
use crate::{record_wo_desc as record, tree};

#[test]
fn k80_simple() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let sequences = Sequences::new(vec![
        record!("A", b"CTATATATAC"),
        record!("B", b"ATATATATAA"),
        record!("C", b"TTATATATAT"),
        record!("D", b"TTATATATAT"),
    ]);
    let info = PIB::build_from_objects(sequences, tree).unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_logl);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_given() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());
    assert_relative_eq!(o.final_logl, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_logl);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_nj() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from an NJ tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::new(fldr.join("K80.fasta")).build().unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());
    assert_relative_eq!(o.final_logl, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_logl);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_vs_phyml() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());

    let c = SCB::new(jc69.clone(), o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_logl);
    assert_relative_eq!(c.cost(), o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();

    let phyml_c = SCB::new(jc69, phyml_info.clone()).build().unwrap();
    assert_relative_eq!(o.final_logl, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.node(&tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn k80_sim_data_vs_phyml_wrong_start() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    // when starting from a wrong tree
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();
    let phyml_c = SCB::new(jc69.clone(), phyml_info.clone()).build().unwrap();

    assert_relative_eq!(o.final_logl, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn wag_no_gaps_vs_phyml_given_tree_start() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let true_tree_file = fldr.join("true_tree.newick");
    let tree_file = fldr.join("jati_wag_nogap.newick");

    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::with_attrs(seq_file.clone(), true_tree_file.clone())
        .build()
        .unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let unopt_logl = c.cost();

    let result =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let o = TopologyOptimiser::new(c.clone()).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            assert_eq!(o.initial_logl, unopt_logl);
            assert_eq!(o.final_logl, o.cost.cost());
            assert!(write_newick_to_file(&[o.cost.tree().clone()], tree_file.clone()).is_ok());
            o.cost.info
        };

    let logl = SCB::new(model.clone(), result.clone())
        .build()
        .unwrap()
        .cost();
    assert!(logl >= unopt_logl);

    let phyml_result = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(model, phyml_result.clone())
        .build()
        .unwrap()
        .cost();

    // Compare tree and logl to PhyML output
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-2);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e-5);
}

#[test]
fn wag_no_gaps_vs_phyml_nj_tree_start() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps starting from an NJ tree
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");

    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::new(seq_file.clone()).build().unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let unopt_logl = c.cost();

    let result =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let o = TopologyOptimiser::new(c).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            assert_eq!(o.initial_logl, unopt_logl);
            assert_eq!(o.final_logl, o.cost.cost());
            assert!(write_newick_to_file(&[o.cost.tree().clone()], tree_file.clone()).is_ok());
            o.cost.info
        };

    let logl = SCB::new(model.clone(), result.clone())
        .build()
        .unwrap()
        .cost();
    assert!(logl >= unopt_logl);

    let phyml_result = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(model.clone(), phyml_result.clone())
        .build()
        .unwrap()
        .cost();

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-3);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e-3);
}

#[test]
fn pip_vs_subst_dna_tree() {
    // Check that optimisation on k80 data under PIP and substitution model produces similar trees
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0]).unwrap();
    let k80_res = TopologyOptimiser::new(SCB::new(k80.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    let pip = PIPModel::<K80>::new(&[], &[0.5, 0.4, 4.0]).unwrap();
    let pip_res = TopologyOptimiser::new(PIPCB::new(pip.clone(), info).build().unwrap())
        .run()
        .unwrap();

    // Tree topologies under PIP+K80 and K80 should match
    assert_eq!(pip_res.cost.tree().robinson_foulds(k80_res.cost.tree()), 0);

    // Check that likelihoods under substitution model are similar for both trees
    // but reoptimise branch lengths for PIP tree because they are not comparable
    let k80_pip_tree_res = BranchOptimiser::new(SCB::new(k80, pip_res.cost.info).build().unwrap())
        .run()
        .unwrap();
    assert_relative_eq!(
        k80_res.final_logl,
        k80_pip_tree_res.final_logl,
        epsilon = 1e-3
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    let pip_k80_tree_res =
        BranchOptimiser::new(PIPCB::new(pip, k80_res.cost.info).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        pip_res.final_logl,
        pip_k80_tree_res.final_logl,
        epsilon = 1e-6
    );
}

#[test]
fn wag_nogaps_pip_vs_subst_tree_nj_start() {
    // ~330s
    // Check that optimisation on protein data under WAG produces similar trees for PIP and substitution model
    // on sequences without gaps
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let pip_tree_file = fldr.join("jati_pip_nogap_pip_vs_wag.newick");
    let wag_tree_file = fldr.join("jati_wag_nogap_pip_vs_wag.newick");

    let info = PIB::new(seq_file.clone()).build().unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[50.0, 0.1]).unwrap();
    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let c_pip = PIPCB::new(pip.clone(), info.clone()).build().unwrap();
    let c_wag = SCB::new(wag.clone(), info.clone()).build().unwrap();

    let unopt_pip_logl = c_pip.cost();
    let result_pip =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), pip_tree_file.clone()).build() {
            precomputed
        } else {
            let o = TopologyOptimiser::new(c_pip).run().unwrap();
            assert!(o.final_logl >= unopt_pip_logl);
            assert!(write_newick_to_file(&[o.cost.tree().clone()], pip_tree_file.clone()).is_ok());
            o.cost.info
        };

    let unopt_wag_logl = c_wag.cost();
    let result_wag =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), wag_tree_file.clone()).build() {
            precomputed
        } else {
            let o = TopologyOptimiser::new(c_wag).run().unwrap();
            assert!(o.final_logl >= unopt_wag_logl);
            assert!(write_newick_to_file(&[o.cost.tree().clone()], wag_tree_file.clone()).is_ok());
            o.cost.info
        };

    // Compare tree created with a substitution model to the one with PIP
    assert_eq!(result_pip.tree.robinson_foulds(&result_wag.tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let wag_pip_tree_res =
        BranchOptimiser::new(SCB::new(wag.clone(), result_pip.clone()).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        SCB::new(wag, result_wag.clone()).build().unwrap().cost(),
        wag_pip_tree_res.final_logl,
        epsilon = 1e-3
    );

    // Check that the likelihoods under the same model are similar for both trees
    let pip_wag_tree_res =
        BranchOptimiser::new(PIPCB::new(pip.clone(), result_wag.clone()).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        PIPCB::new(pip.clone(), result_pip.clone())
            .build()
            .unwrap()
            .cost(),
        pip_wag_tree_res.final_logl,
        epsilon = 1e-3
    );

    let new_pip_c = PIPCB::new(pip, result_pip).build().unwrap();
    let o2_subst = BranchOptimiser::new(new_pip_c.clone()).run().unwrap();

    assert_relative_eq!(new_pip_c.cost(), o2_subst.final_logl, epsilon = 1e-4);
}

#[test]
fn protein_pip_optimise_model_tree() {
    // ~430s
    // Check that tree optimisation under PIP has a better likelihood when the model is also optimised
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");

    let pip = PIPModel::<WAG>::new(&[], &[1.4, 0.5]).unwrap();

    let tree_file = fldr.join("jati_pip_nj_start.newick");
    let result =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let info = PIB::new(seq_file.clone()).build().unwrap();
            let pip_c = PIPCB::new(pip.clone(), info).build().unwrap();
            let unopt_logl = pip_c.cost();
            let o = TopologyOptimiser::new(pip_c).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            assert!(write_newick_to_file(&[o.cost.tree().clone()], tree_file.clone()).is_ok());
            o.cost.info
        };

    let tree_file = fldr.join("jati_pip_nj_start_model_opt.newick");
    let result_model_opt =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let info = PIB::new(seq_file.clone()).build().unwrap();
            let pip_c = PIPCB::new(pip.clone(), info).build().unwrap();
            let unopt_logl = pip_c.cost();
            let o = ModelOptimiser::new(pip_c, Empirical).run().unwrap();
            let model_opt_logl = o.final_logl;
            assert!(model_opt_logl >= unopt_logl);
            let o = TopologyOptimiser::new(o.cost).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            assert!(write_newick_to_file(&[o.cost.tree().clone()], tree_file.clone()).is_ok());
            o.cost.info
        };

    let pip_opt = PIPModel::<WAG>::new(&[], &[49.56941, 0.09352]).unwrap();
    assert_eq!(result_model_opt.tree.robinson_foulds(&result.tree), 0);

    let pip_opt_new_res_logl = PIPCB::new(pip_opt.clone(), result_model_opt.clone())
        .build()
        .unwrap()
        .cost();
    let pip_opt_old_res_logl = PIPCB::new(pip_opt, result.clone()).build().unwrap().cost();
    let pip_old_res_logl = PIPCB::new(pip.clone(), result).build().unwrap().cost();
    let pip_new_res_logl = PIPCB::new(pip, result_model_opt.clone())
        .build()
        .unwrap()
        .cost();

    assert!(pip_opt_new_res_logl > pip_old_res_logl);
    assert!(pip_opt_new_res_logl > pip_opt_old_res_logl);
    assert!(pip_opt_new_res_logl > pip_new_res_logl);
}

#[test]
fn protein_wag_vs_phyml_empirical_freqs() {
    // ~110s
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using empirical frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_empirical.newick");

    let info = PIB::new(seq_file.clone()).build().unwrap();
    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let c_wag = SCB::new(wag.clone(), info.clone()).build().unwrap();

    let unopt_logl = c_wag.cost();

    let o = ModelOptimiser::new(c_wag, Empirical).run().unwrap();
    let model_opt_logl = o.final_logl;
    assert!(model_opt_logl >= unopt_logl);
    let wag_opt = o.cost.model;

    let result =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let info = PIB::new(seq_file.clone()).build().unwrap();
            let c_wag_opt = SCB::new(wag_opt.clone(), info).build().unwrap();
            let unopt_logl = c_wag_opt.cost();
            let o = TopologyOptimiser::new(c_wag_opt).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            let result = o.cost.info;
            assert!(write_newick_to_file(&[result.tree.clone()], tree_file.clone()).is_ok());
            result
        };

    let phyml_result = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();

    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-4);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);

    let wag_opt_logl = SCB::new(wag_opt.clone(), result).build().unwrap().cost();
    let wag_opt_phyml_logl = SCB::new(wag_opt, phyml_result).build().unwrap().cost();
    assert_relative_eq!(wag_opt_logl, wag_opt_phyml_logl, epsilon = 1e-5);
    assert_relative_eq!(wag_opt_logl, -5258.79254297163, epsilon = 1e-5);
}

#[test]
fn protein_wag_vs_phyml_fixed_freqs() {
    // ~100s
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using fixed frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_fixed.newick");
    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();

    let result =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            precomputed
        } else {
            let info = PIB::new(seq_file.clone()).build().unwrap();
            let c_wag = SCB::new(wag.clone(), info).build().unwrap();
            let unopt_logl = c_wag.cost();
            let o = TopologyOptimiser::new(c_wag).run().unwrap();
            assert!(o.final_logl >= unopt_logl);
            let result = o.cost.info;
            assert!(write_newick_to_file(&[result.tree.clone()], tree_file.clone()).is_ok());
            result
        };

    let phyml_result = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_fixed.newick"))
        .build()
        .unwrap();
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-4);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);

    let wag_logl = SCB::new(wag.clone(), result).build().unwrap().cost();
    let wag_phyml_logl = SCB::new(wag, phyml_result).build().unwrap().cost();

    assert_relative_eq!(wag_logl, wag_phyml_logl, epsilon = 1e-5);
    assert_relative_eq!(wag_logl, -5295.084233408107, epsilon = 1e-2);
}
