use std::path::Path;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences};
use crate::evolutionary_models::FrequencyOptimisation::Empirical;
#[cfg(feature = "use-precomputed")]
use crate::io::write_newick_to_file;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, ModelOptimiser, TopologyOptimiser};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, SubstModel, SubstitutionCostBuilder as SCB,
};
use crate::{record_wo_desc as record, tree};

#[test]
fn k80_simple() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = Alignment::from_aligned(
        Sequences::new(vec![
            record!("A", b"CTATATATAC"),
            record!("B", b"ATATATATAA"),
            record!("C", b"TTATATATAT"),
            record!("D", b"TTATATATAT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_given() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_nj() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from an NJ tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::new(fldr.join("K80.fasta")).build().unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_vs_phyml() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]);
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(jc69.clone(), o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();

    let phyml_c = SCB::new(jc69, phyml_info.clone()).build().unwrap();
    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

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
    let jc69 = SubstModel::<JC69>::new(&[], &[]);
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c).run().unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();
    let phyml_c = SCB::new(jc69.clone(), phyml_info.clone()).build().unwrap();

    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

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
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_no_gaps_vs_phyml_nj_tree_start() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps starting from an NJ tree
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");

    let wag = SubstModel::<WAG>::new(&[], &[]);
    let start_info = PIB::new(seq_file.clone()).build().unwrap();
    let wag_cost = SCB::new(wag.clone(), start_info).build().unwrap();

    let unopt_logl = wag_cost.cost();
    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let wag_tree_file = fldr.join("jati_wag_nogap_nj_start.newick");
        let (wag_res, final_wag_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), wag_tree_file.clone()).build() {
            let precomputed_cost = SCB::new(wag.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(wag_cost.clone()).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], wag_tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(wag_cost.clone()).run().unwrap();
        let (wag_res, final_wag_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_wag_logl >= unopt_logl);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(wag, phyml_res.clone()).build().unwrap().cost();

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(wag_res.tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(wag_res.tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(final_wag_logl, phyml_logl, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_vs_subst_dna_tree() {
    // Check that optimisation on k80 data under PIP and substitution model produces similar trees
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0]);
    let k80_res = TopologyOptimiser::new(SCB::new(k80.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    let pip = PIPModel::<K80>::new(&[], &[0.5, 0.4, 4.0]);
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
        k80_res.final_cost,
        k80_pip_tree_res.final_cost,
        epsilon = 1e-6
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    let pip_k80_tree_res =
        BranchOptimiser::new(PIPCB::new(pip, k80_res.cost.info).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        pip_res.final_cost,
        pip_k80_tree_res.final_cost,
        epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_nogaps_pip_vs_subst_tree_nj_start() {
    // Check that optimisation on protein data under WAG produces similar trees for PIP and substitution model
    // on sequences without gaps
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let pip = PIPModel::<WAG>::new(&[], &[50.0, 0.1]);
    let wag = SubstModel::<WAG>::new(&[], &[]);
    let pip_cost = PIPCB::new(pip.clone(), start_info.clone()).build().unwrap();
    let wag_cost = SCB::new(wag.clone(), start_info.clone()).build().unwrap();

    let unopt_pip_logl = pip_cost.cost();
    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let pip_tree_file = fldr.join("jati_pip_nogap_pip_vs_wag.newick");
        let (pip_res, final_pip_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), pip_tree_file.clone()).build() {
            let precomputed_cost = PIPCB::new(pip.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(pip_cost.clone()).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], pip_tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(pip_cost.clone()).run().unwrap();
        let (pip_res, final_pip_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_pip_logl >= unopt_pip_logl);

    let unopt_wag_logl = wag_cost.cost();
    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let wag_tree_file = fldr.join("jati_wag_nogap_pip_vs_wag.newick");
        let (wag_res, final_wag_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), wag_tree_file.clone()).build() {
            let precomputed_cost = SCB::new(wag.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(wag_cost.clone()).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], wag_tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(wag_cost.clone()).run().unwrap();
        let (wag_res, final_wag_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_wag_logl >= unopt_wag_logl);

    // Compare tree created with a substitution model to the one with PIP
    assert_eq!(pip_res.tree.robinson_foulds(&wag_res.tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let pip_tree_reopt_wag_logl =
        BranchOptimiser::new(SCB::new(wag.clone(), pip_res.clone()).build().unwrap())
            .run()
            .unwrap()
            .final_cost;
    assert_relative_eq!(final_wag_logl, pip_tree_reopt_wag_logl, epsilon = 1e-5);

    // Check that the likelihoods under the same model are similar for both trees
    let wag_tree_reopt_pip_logl =
        BranchOptimiser::new(PIPCB::new(pip.clone(), wag_res.clone()).build().unwrap())
            .run()
            .unwrap()
            .final_cost;
    assert_relative_eq!(final_pip_logl, wag_tree_reopt_pip_logl, epsilon = 1e-5);

    // Just a random check that reestimating branch lengths makes no difference
    let new_pip_cost = PIPCB::new(pip, pip_res.clone()).build().unwrap();
    let reopt_res = BranchOptimiser::new(new_pip_cost.clone()).run().unwrap();
    assert_relative_eq!(new_pip_cost.cost(), reopt_res.final_cost, epsilon = 1e-5);
    assert_relative_eq!(
        pip_res.tree.height,
        reopt_res.cost.info.tree.height,
        epsilon = 1e-5
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_wag_optimise_model_tree() {
    // Check that tree optimisation under PIP has a better likelihood when the model is also optimised
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let pip = PIPModel::<WAG>::new(&[], &[1.4, 0.5]);
    let pip_cost = PIPCB::new(pip.clone(), start_info).build().unwrap();
    let unopt_logl = pip_cost.cost();

    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let tree_file = fldr.join("jati_pip_nj_start.newick");
        let (res, model_unopt_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            let precomputed_cost = PIPCB::new(pip.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(pip_cost.clone()).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(pip_cost.clone()).run().unwrap();
        let (res, model_unopt_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(model_unopt_logl >= unopt_logl);

    // Optimise model parameters
    let o = ModelOptimiser::new(pip_cost, Empirical).run().unwrap();
    let model_opt_logl = o.final_cost;
    let pip_opt = o.cost.model;
    let pip_opt_cost = PIPCB::new(pip_opt.clone(), o.cost.info).build().unwrap();

    assert!(model_opt_logl >= unopt_logl);
    assert!(model_opt_logl >= model_unopt_logl);

    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let tree_file = fldr.join("jati_pip_nj_start_model_opt.newick");
        let (model_opt_res, final_model_opt_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            let precomputed_cost = PIPCB::new(pip_opt.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(pip_opt_cost).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(pip_opt_cost).run().unwrap();
        let (model_opt_res, final_model_opt_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_model_opt_logl >= model_opt_logl);
    assert!(final_model_opt_logl >= unopt_logl);

    assert_eq!(model_opt_res.tree.robinson_foulds(&res.tree), 0);

    assert!(final_model_opt_logl > model_unopt_logl);
    assert!(final_model_opt_logl > PIPCB::new(pip_opt, res.clone()).build().unwrap().cost());
    assert!(
        final_model_opt_logl
            > PIPCB::new(pip, model_opt_res.clone())
                .build()
                .unwrap()
                .cost()
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_empirical_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using empirical frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let wag = SubstModel::<WAG>::new(&[], &[]);
    let wag_cost = SCB::new(wag.clone(), start_info).build().unwrap();
    let unopt_logl = wag_cost.cost();

    let o = ModelOptimiser::new(wag_cost, Empirical).run().unwrap();
    let model_opt_logl = o.final_cost;
    assert!(model_opt_logl >= unopt_logl);
    let wag_opt = o.cost.model;
    let wag_opt_cost = SCB::new(wag_opt.clone(), o.cost.info).build().unwrap();

    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let tree_file = fldr.join("jati_wag_empirical.newick");
        let (res, final_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            let precomputed_cost = SCB::new(wag_opt.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(wag_opt_cost).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(wag_opt_cost).run().unwrap();
        let (res, final_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_logl >= unopt_logl);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();

    assert_relative_eq!(res.tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(res.tree.robinson_foulds(&phyml_res.tree), 0);

    let wag_opt_phyml_logl = SCB::new(wag_opt, phyml_res).build().unwrap().cost();
    assert_relative_eq!(final_logl, wag_opt_phyml_logl, epsilon = 1e-5);
    assert_relative_eq!(final_logl, -5258.79254297163, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_wag_vs_phyml_empirical_freqs() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");

    let start_info = PIB::new(seq_file.clone()).build().unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[1.0, 2.0]);
    // Use empirical frequencies and optimise lambda and mu for PIP
    let o = ModelOptimiser::new(
        PIPCB::new(pip.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();

    let model_opt_logl = o.final_cost;
    let pip_opt = o.cost.model;
    let pip_opt_cost = PIPCB::new(pip_opt.clone(), start_info).build().unwrap();

    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let tree_file = fldr.join("jati_pip_wag_empirical.newick");
        let (res, final_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            let precomputed_cost = PIPCB::new(pip_opt.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(pip_opt_cost).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(pip_opt_cost).run().unwrap();
        let (res, final_logl) = (o.cost.info, o.final_cost);
    }
    }

    // Optimised tree should be better than the starting tree
    assert!(final_logl >= model_opt_logl);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();
    // Reoptimise branch lengths for the phyml tree because they are not comparable with PIP
    let phyml_brlen_opt = BranchOptimiser::new(PIPCB::new(pip_opt, phyml_res).build().unwrap())
        .run()
        .unwrap();

    // Check that our tree is better than phyml
    assert!(final_logl > phyml_brlen_opt.final_cost);
    assert_relative_eq!(
        res.tree.height,
        phyml_brlen_opt.cost.info.tree.height,
        epsilon = 1e-2
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_fixed_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using fixed frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let wag = SubstModel::<WAG>::new(&[], &[]);
    let start_info = PIB::new(seq_file.clone()).build().unwrap();

    let wag_cost = SCB::new(wag.clone(), start_info).build().unwrap();
    let unopt_logl = wag_cost.cost();

    cfg_if::cfg_if! {
    if #[cfg(feature = "use-precomputed")]{
        let tree_file = fldr.join("jati_wag_fixed.newick");
        let (res, final_logl) =
        if let Ok(precomputed) = PIB::with_attrs(seq_file.clone(), tree_file.clone()).build() {
            let precomputed_cost = SCB::new(wag.clone(), precomputed.clone()).build().unwrap().cost();
            (precomputed, precomputed_cost)
        } else {
            let o = TopologyOptimiser::new(wag_cost).run().unwrap();
            assert!(write_newick_to_file(&[o.cost.info.tree.clone()], tree_file).is_ok());
            (o.cost.info, o.final_cost)
        };
    } else {
        let o = TopologyOptimiser::new(wag_cost).run().unwrap();
        let (res, final_logl) = (o.cost.info, o.final_cost);
    }
    }
    assert!(final_logl >= unopt_logl);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_fixed.newick"))
        .build()
        .unwrap();

    assert_relative_eq!(res.tree.height, phyml_res.tree.height, epsilon = 1e-4);
    assert_eq!(res.tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(
        final_logl,
        SCB::new(wag, phyml_res).build().unwrap().cost(),
        epsilon = 1e-5
    );

    assert_relative_eq!(final_logl, -5295.08423, epsilon = 1e-3);
}
