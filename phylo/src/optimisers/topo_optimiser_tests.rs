use std::path::Path;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::{DNAModelType::*, EvoModel, ProteinModelType::*};
use crate::io::write_newick_to_file;
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{
    BranchOptimiser, EvoModelOptimiser, FrequencyOptimisation::Empirical, ModelOptimiser,
    PhyloOptimiser, TopologyOptimiser,
};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};
use crate::tree::tree_parser::from_newick_string;

#[test]
fn k80_topo_optimisation() {
    let tree =
        from_newick_string("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);").unwrap()[0].clone();
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
        Record::with_attrs("D", None, b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(&info, false);
    let o = TopologyOptimiser::new(&k80, &info).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i, true));
}

#[test]
fn k80_sim_topo_optimisation_from_nj() {
    let info = PhyloInfoBuilder::new(Path::new("./data/sim/K80/K80.fasta").to_path_buf())
        .build()
        .unwrap();
    let k80 = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = k80.cost(&info, false);
    let o = TopologyOptimiser::new(&k80, &info).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, k80.cost(&o.i, true));
    assert_relative_eq!(o.final_logl, -4060.91964, epsilon = 1e-5);
}

#[test]
fn topology_opt_simulated_from_tree() {
    let fldr = Path::new("./data/sim/K80");
    let info = PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = jc69.cost(&info, false);
    let o = TopologyOptimiser::new(&jc69, &info).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, jc69.cost(&o.i, true), epsilon = 1e-5);

    let phyml_info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("phyml_tree.newick"))
            .build()
            .unwrap();

    assert_relative_eq!(o.final_logl, jc69.cost(&phyml_info, true), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.node(&o.i.tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-5);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn topology_opt_simulated_from_wrong_tree() {
    let fldr = Path::new("./data/sim/K80");
    let info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("../wrong_tree.newick"))
            .build()
            .unwrap();
    let jc69 = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = jc69.cost(&info, false);
    let o = TopologyOptimiser::new(&jc69, &info).run().unwrap();

    assert!(o.final_logl >= unopt_logl);
    assert_relative_eq!(o.final_logl, jc69.cost(&o.i, true), epsilon = 1e-5);

    let phyml_info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("phyml_tree.newick"))
            .build()
            .unwrap();

    assert_relative_eq!(o.final_logl, jc69.cost(&phyml_info, true), epsilon = 1e-5);
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            o.i.tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    // Slightly different rooting leads to a different breakdown of branches,
    // but sum is still same
    assert_relative_eq!(
        o.i.tree.by_id("Human").blen
            + o.i
                .tree
                .node(&(o.i.tree.by_id("Chimpanzee")).parent.unwrap())
                .blen,
        phyml_info.tree.by_id("Human").blen,
        epsilon = 1e-5
    );
    assert_relative_eq!(o.i.tree.height, phyml_info.tree.height, epsilon = 1e-4);
    assert_eq!(o.i.tree.robinson_foulds(&phyml_info.tree), 0);
}

#[test]
fn protein_topology_optimisation_given_tree_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let true_tree_file = fldr.join("true_tree.newick");
    let tree_file = fldr.join("jati_wag_nogap.newick");

    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let info = PhyloInfoBuilder::with_attrs(seq_file.clone(), true_tree_file.clone())
        .build()
        .unwrap();
    let unopt_logl = model.cost(&info, false);

    let result = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let o = TopologyOptimiser::new(&model, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], tree_file.clone()).is_ok());
        o.i
    };

    let logl = model.cost(&result, true);
    assert!(logl >= unopt_logl);

    let phyml_result =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
            .build()
            .unwrap();
    let phyml_logl = model.cost(&phyml_result, true);

    // Compare tree and logl to PhyML output
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-4);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e-5);
}

#[test]
fn protein_topology_optimisation_nj_start() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");

    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
    let unopt_logl = model.cost(&info, false);

    let result = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let o = TopologyOptimiser::new(&model, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], tree_file.clone()).is_ok());
        o.i
    };

    let logl = model.cost(&result, true);
    assert!(logl >= unopt_logl);

    let phyml_result =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
            .build()
            .unwrap();
    let phyml_logl = model.cost(&phyml_result, true);

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-3);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
    assert_relative_eq!(logl, phyml_logl, epsilon = 1e-5);
}

#[test]
fn pip_vs_subst_dna_tree() {
    let fldr = Path::new("./data/sim/K80");
    let info =
        PhyloInfoBuilder::with_attrs(fldr.join("K80.fasta"), fldr.join("../wrong_tree.newick"))
            .build()
            .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let k80_opt_res = TopologyOptimiser::new(&model, &info).run().unwrap();

    let pip = PIPModel::<DNASubstModel>::new(K80, &[0.5, 0.4, 4.0, 1.0]).unwrap();
    let pip_opt_res = TopologyOptimiser::new(&pip, &info).run().unwrap();

    // Tree topologies under PIP+K80 and K80 should match
    assert_eq!(pip_opt_res.i.tree.robinson_foulds(&k80_opt_res.i.tree), 0);

    // Check that likelihoods under substitution model are similar for both trees
    // but reoptimise branch lengths for PIP tree because they are not comparable
    let o_pip_blen = BranchOptimiser::new(&model, &pip_opt_res.i).run().unwrap();
    assert_relative_eq!(
        k80_opt_res.final_logl,
        o_pip_blen.final_logl,
        epsilon = 1e-3
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    let o_k80_blen = BranchOptimiser::new(&pip, &k80_opt_res.i).run().unwrap();
    assert_relative_eq!(
        pip_opt_res.final_logl,
        o_k80_blen.final_logl,
        epsilon = 1e-6
    );
}

#[test]
fn pip_vs_subst_protein_tree_nogaps() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let pip_tree_file = fldr.join("jati_pip_nogap_pip_vs_wag.newick");
    let wag_tree_file = fldr.join("jati_wag_nogap_pip_vs_wag.newick");

    let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
    let pip = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    let wag = ProteinSubstModel::new(WAG, &[]).unwrap();

    let unopt_pip_logl = pip.cost(&info, true);
    let result_pip = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), pip_tree_file.clone()).build()
    {
        precomputed
    } else {
        let o = TopologyOptimiser::new(&pip, &info).run().unwrap();
        assert!(o.final_logl >= unopt_pip_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], pip_tree_file.clone()).is_ok());
        o.i
    };

    let unopt_wag_logl = wag.cost(&info, true);
    let result_wag = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), wag_tree_file.clone()).build()
    {
        precomputed
    } else {
        let o = TopologyOptimiser::new(&wag, &info).run().unwrap();
        assert!(o.final_logl >= unopt_wag_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], wag_tree_file.clone()).is_ok());
        o.i
    };

    // Compare tree created with a substitution model to the one with PIP
    assert_eq!(result_pip.tree.robinson_foulds(&result_wag.tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let pip_reopt_result = BranchOptimiser::new(&wag, &result_pip).run().unwrap();
    assert_relative_eq!(
        wag.cost(&result_wag, true),
        pip_reopt_result.final_logl,
        epsilon = 1e-5
    );

    // Check that the likelihoods under the same model are similar for both trees
    let pip_wag = PIPModel::<ProteinSubstModel>::new(WAG, &[50.0, 0.1]).unwrap();
    let o2_subst = BranchOptimiser::new(&pip_wag, &result_wag).run().unwrap();

    assert_relative_eq!(
        pip_wag.cost(&result_pip, true),
        o2_subst.final_logl,
        epsilon = 1e-5
    );
}

#[test]
fn protein_optimise_model_tree() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");

    let pip = PIPModel::<ProteinSubstModel>::new(WAG, &[1.4, 0.5]).unwrap();

    let tree_file = fldr.join("jati_pip_nj_start.newick");
    let result = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
        let unopt_logl = pip.cost(&info, true);
        let o = TopologyOptimiser::new(&pip, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], tree_file.clone()).is_ok());
        o.i
    };

    let tree_file = fldr.join("jati_pip_nj_start_model_opt.newick");
    let result_model_opt = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
        let unopt_logl = pip.cost(&info, true);
        let o = ModelOptimiser::new(&pip, &info, Empirical).run().unwrap();
        let model_opt_logl = o.final_logl;
        assert!(model_opt_logl >= unopt_logl);
        let o = TopologyOptimiser::new(&o.model, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        assert!(write_newick_to_file(&[o.i.tree.clone()], tree_file.clone()).is_ok());
        o.i
    };

    let pip_opt = PIPModel::<ProteinSubstModel>::new(WAG, &[49.56941, 0.09352]).unwrap();

    assert_eq!(result_model_opt.tree.robinson_foulds(&result.tree), 0);
    assert!(pip_opt.cost(&result_model_opt, true) > pip.cost(&result, true));
    assert!(pip_opt.cost(&result_model_opt, true) > pip_opt.cost(&result, true));
    assert!(pip_opt.cost(&result_model_opt, true) > pip.cost(&result, true));
}

#[test]
fn protein_wag_vs_phyml_empirical_freqs() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_empirical.newick");

    let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let unopt_logl = model.cost(&info, false);

    let o = ModelOptimiser::new(&model, &info, Empirical).run().unwrap();
    let model_opt_logl = o.final_logl;
    assert!(model_opt_logl >= unopt_logl);
    let model = o.model;

    let result = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
        let unopt_logl = model.cost(&info, false);
        let o = TopologyOptimiser::new(&model, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        let result = o.i;
        // The above ran in 394.21s
        assert!(write_newick_to_file(&[result.tree.clone()], tree_file.clone()).is_ok());
        result
    };

    let phyml_result =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
            .build()
            .unwrap();

    assert_relative_eq!(
        model.cost(&result, true),
        model.cost(&phyml_result, true),
        epsilon = 1e-5
    );
    assert_relative_eq!(model.cost(&result, true), -5258.79254297163, epsilon = 1e-5);
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-4);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
}

#[test]
fn protein_wag_vs_phyml_fixed_freqs() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_wag_fixed.newick");
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();

    let result = if let Ok(precomputed) =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), tree_file.clone()).build()
    {
        precomputed
    } else {
        let info = PhyloInfoBuilder::new(seq_file.clone()).build().unwrap();
        let unopt_logl = model.cost(&info, false);
        let o = TopologyOptimiser::new(&model, &info).run().unwrap();
        assert!(o.final_logl >= unopt_logl);
        let result = o.i;
        // The above ran in 372.59s
        assert!(write_newick_to_file(&[result.tree.clone()], tree_file.clone()).is_ok());
        result
    };

    let phyml_result =
        PhyloInfoBuilder::with_attrs(seq_file.clone(), fldr.join("phyml_wag_fixed.newick"))
            .build()
            .unwrap();

    assert_relative_eq!(
        model.cost(&result, true),
        model.cost(&phyml_result, true),
        epsilon = 1e-5
    );
    assert_relative_eq!(
        model.cost(&result, true),
        -5295.084233408107,
        epsilon = 1e-2
    );
    assert_relative_eq!(result.tree.height, phyml_result.tree.height, epsilon = 1e-4);
    assert_eq!(result.tree.robinson_foulds(&phyml_result.tree), 0);
}
