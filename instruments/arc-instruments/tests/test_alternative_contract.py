"""Counterexamples derived from the measurement and custody contract, not fitted worlds.

This file arrived with a partial carry of a parallel branch and held that carry's subjects only. Four
of the subjects it was missing have since landed from the other branch, in stronger forms, and are
covered by the tests beside this file rather than by it: repeat-read precision in
`test_ladder_sampling.py`, the prediction's own uncertainty inside the comparison and the coupling
domain in `test_p5_final_comparison.py`, portable code identity and the verifier binding in
`test_custody.py`. Several checks here are therefore the second reading of a repair rather than its
only one, which is the intended state.

Two subjects are still absent from the tree, and are named here rather than left to be discovered.
`loglog_slope` still floors its scores at 1e-9 before taking the logarithm, so a checkpoint reading of
zero yields a finite fabricated slope instead of a missing one. `CheckpointStore` still saves an
artefact without a digest sidecar beside it, so a checkpoint altered on disk between the save and the
load is not detected by the store itself. Both change what runs decide, which is why neither was
taken on the way past: each needs its own reading and its own regenerated battery.
"""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from arc_runner import manifest as M, evidence, observation as OBS, p5, p16, trajectory as T, code_domain as CD
from arc_runner.p16_contract import Interval as I, UnitPrediction as U, Contract, Observation as O, LineObservation as Line, adjudicate, equivalent, log_ratio_elasticity


def contract():
    # Fixture values are examples for this proposed software contract, not new preregistered thresholds.
    units=(U('above','above',I(2.59,2.61),I(-.31,-.29),I(25,27),64),
           U('below','below',I(1.39,1.41),I(.29,.31),None,64),
           U('sham','sham',I(1.99,2.01),I(-.01,.01),None,64),
           U('baseline','baseline',I(1.99,2.01),I(-.01,.01),None,64))
    return Contract('a'*64,units,I(-.51,-.49),I(1.99,2.01),.1,.1,.15,.1,4,.05,.5,'joint-EIV-v1')


def observations(cfg):
    return {u.unit_id:O(u.alpha,u.delta,u.event_depth,64,False,True,cfg.interval_method_id) for u in cfg.units}


def line(cfg): return Line(cfg.line_slope,cfg.line_zero,cfg.interval_method_id,True)


def test_balanced_ratio_needs_log_depth_and_is_invariant_to_units():
    R=np.array([1,2,4,8,16]); Q=3*R**.8; W=5*R**1.1
    assert log_ratio_elasticity(R,Q,W)==pytest.approx(-.3)
    assert log_ratio_elasticity(10*R,17*Q,.2*W)==pytest.approx(-.3)
    # A decreasing positive Delta is still positive; regressing Delta on rounds tests a different quantity.
    assert equivalent(I(.2,.3),.1)=='ADVERSE'
    with pytest.raises(ValueError): log_ratio_elasticity([0,1,2],[1,1,1],[1,1,1])


def test_correct_zero_wrong_slope_does_not_pass():
    c=contract(); o=observations(c)
    correct=adjudicate(c,o,line(c))
    assert correct['contract_result']=='MATCHES CANDIDATE CONTRACT'
    assert correct['empirical_verdict']=='NOT TESTED'
    bad=replace(line(c),slope=I(-.011,-.009))
    assert adjudicate(c,o,bad)['contract_result']=='ADVERSE TO CANDIDATE CONTRACT'


@pytest.mark.parametrize('change',[{'event_depth':I(43,45)}, {'delta':I(.2,.3)}])
def test_late_or_opposite_direction_is_adverse(change):
    c=contract();o=observations(c);o['above']=replace(o['above'],**change)
    assert adjudicate(c,o,line(c))['contract_result']=='ADVERSE TO CANDIDATE CONTRACT'


@pytest.mark.parametrize('change',[{'event_depth':None}, {'delta':I(-.5,.1)}, {'alpha':None},
    {'observed_horizon':40}, {'censored':True}, {'measurement_valid':False}, {'interval_method_id':'other'}])
def test_no_alarm_wide_interval_missing_assay_and_censoring_stay_unresolved(change):
    c=contract();o=observations(c);o['above']=replace(o['above'],**change)
    assert adjudicate(c,o,line(c))['contract_result']=='UNRESOLVED'


def test_controls_have_separate_denominators():
    c=contract();o=observations(c);o['sham']=replace(o['sham'],delta=I(-.4,-.3))
    assert adjudicate(c,o,line(c))['contract_result']=='NOT SPECIFIC'


def test_missing_units_are_not_silently_removed():
    c=contract();o=observations(c);o.pop('above')
    got=adjudicate(c,o,line(c))
    assert got['contract_result']=='UNRESOLVED' and got['groups']['above']['assigned']==1
    o['invented']=o['below']
    with pytest.raises(ValueError): adjudicate(c,o,line(c))


def test_line_needs_realised_alpha_and_a_localised_interval():
    c=contract();o=observations(c)
    assert adjudicate(c,o,replace(line(c),uses_independent_measured_alpha=False))['contract_result']=='UNRESOLVED'
    assert adjudicate(c,o,replace(line(c),zero=I(1.2,2.8)))['contract_result']=='UNRESOLVED'


@pytest.mark.parametrize('d,expected',[(I(-.09,.09),'MATCHED'),(I(-.1,.09),'UNRESOLVED'),
    (I(.1,.2),'UNRESOLVED'),(I(.10001,.2),'ADVERSE'),(I(-.2,-.10001),'ADVERSE')])
def test_strict_interval_boundaries(d,expected): assert equivalent(d,.1)==expected


def test_zero_retention_carries_no_context():
    # M-11: the retention floor is gone, so the no-retention arm really carries nothing.
    assert T.retain({'text':'a\nb'},0,np.random.default_rng(0))['text']==''


def test_beta_one_is_exponential_and_finite_window_prediction_matches_same_estimator():
    assert p5.calibrate_rate([1,2,3],np.exp(.2*np.array([1,2,3])),1)==pytest.approx(.2)
    R=np.array([1,2,4,8,16,32]); C=(1+.05*(R-1))**2
    expected=np.polyfit(np.log(R),np.log(C),1)[0]
    assert p5.predicted_exponent(.5,.1,1,1,32,R)==pytest.approx(expected)
    assert expected<1


def demo():
    cfg=p16.P16Config(systems_per_arm=1,horizon=30)
    return p16.run_p16(p16.mock_margin_source(cfg,2),cfg,7,'mock-ladder','mock')


def test_every_development_entry_point_refuses_empirical_promotion():
    """A development run may not be promoted to an empirical one by any door.

    THE READINGS CARRY BOTH NAMES AND THE STATUS SAYS WHICH THEY ARE. An earlier form of this check
    asked that the key `verdicts` be absent, on a tree where nothing could ever be scored and the
    word was therefore always wrong. This tree has a deciding mode, reached only through an external
    anchor receipt, a named unseen-material attestation and a bound verifier, so the word is right
    for the one mode that clears them and wrong for every other. The export carries both names for
    one object and states the evidence status beside them, which is what a reader needs: a key name
    alone never says what a reading is.
    """
    run=demo()
    assert run['empirical_verdict']=='NOT TESTED'
    assert run['evidence_status']=='SIMULATION/DEVELOPMENT ONLY'
    assert run['diagnostics'] is run['verdicts']          # both names, one object
    # A demonstration is refused at proposition level, and inventing a public registration string in
    # the manifest does not move it: the anchor that counts is the receipt inside the sealed digest,
    # and this field is not read by the gate at all.
    with pytest.raises(M.NotScoreable): M.require_scoreable(run['manifest'])
    run['manifest']['anchor_identifier']='invented-public-registration'
    with pytest.raises(M.NotScoreable): M.require_scoreable(run['manifest'])
    # A confirmatory titration refuses BEFORE the first call to its source, so a run that cannot be
    # read has not been paid for. A simulated source is refused for being simulated; a real endpoint
    # is refused because the service and burden measurements a real titration needs are not released.
    called=[]
    with pytest.raises(OBS.ObservationRefusal):
        p16.run_p16(lambda *args:called.append(args),p16.P16Config(),1,'x','real',mode='confirmatory')
    assert not called
    class _Remote:
        uses_remote_endpoint=True
        def __call__(self,*args): called.append(args); return 0.0
    with pytest.raises(M.InstrumentNotReleased):
        p16.run_p16(_Remote(),p16.P16Config(),1,'x','real')
    assert not called


def test_manifest_checks_prediction_config_mode_and_code_changes():
    run=demo();m=run['manifest']
    for alter in ('config','prediction','code','mode'):
        changed=deepcopy(m)
        if alter=='config': changed['config']['seed']+=1
        if alter=='prediction': changed['sealed_predictions']['line']['zero']=3
        if alter=='code': changed['code_sha256']='b'*64
        if alter=='mode': changed['mode']='pilot'
        with pytest.raises((ValueError,M.InstrumentNotReleased)): M.require_integrity(changed)


def test_evidence_restart_replay_and_external_root(tmp_path):
    run=demo(); target=tmp_path/'run'
    receipt=evidence.write_bundle(run,target)
    assert evidence.replay_bundle(target,expected_root=receipt['root_sha256'])['reproduced']
    saved=json.loads((target/'run.json').read_text(),parse_constant=lambda x:pytest.fail(x))
    # The summaries and the series are both preserved, in the two places the run returns them: the
    # arm carries what a printed summary carries, and the measured series travels beside it keyed to
    # the arm and the replicate it belongs to.
    assert len(saved['arms'])==7 and all('margin' not in a for a in saved['arms'])
    assert len(saved['arm_series'])==7 and all(len(a['margin'])==30 for a in saved['arm_series'])
    assert len({(a['arm'],a['replicate_id']) for a in saved['arms']})==7
    assert {(a['arm'],a['replicate_id']) for a in saved['arm_series']} \
        == {(a['arm'],a['replicate_id']) for a in saved['arms']}
    with pytest.raises(FileExistsError): evidence.write_bundle(run,target)
    with pytest.raises(ValueError): evidence.verify_bundle(target,expected_root='0'*64)
    raw=(target/'run.json').read_bytes();(target/'run.json').write_bytes(raw+b' ')
    with pytest.raises(ValueError): evidence.verify_bundle(target)


def test_replay_detects_rewritten_terminal_summary_even_with_fresh_local_receipt(tmp_path):
    run=demo();run['diagnostics']['line_zero_fitted']=99
    evidence.write_bundle(run,tmp_path/'run')
    with pytest.raises(ValueError,match='reproduce'): evidence.replay_bundle(tmp_path/'run')


def test_bundle_rejects_extra_files_and_symlinks(tmp_path):
    evidence.write_bundle(demo(),tmp_path/'run')
    (tmp_path/'run'/'unexpected').write_text('x')
    with pytest.raises(ValueError): evidence.verify_bundle(tmp_path/'run')
    (tmp_path/'alias').symlink_to(tmp_path/'run',target_is_directory=True)
    with pytest.raises(ValueError): evidence.verify_bundle(tmp_path/'alias')


def test_state_target_must_be_an_integer():
    # M-26: 3.7 and 3.2 must not both name state-3 and collide on one artefact.
    with pytest.raises(ValueError): CD.state_name(1.2)


def test_early_successful_process_exit_does_not_pass_hidden_checks():
    task=CD.Task('x','Add','def f(): ...',('assert f() == 2',))
    assert not CD.subprocess_verifier(timeout_s=2)('raise SystemExit(0)',task)


def test_p5_headroom_failure_stays_in_assigned_denominator():
    cfg=p5.P5Config()
    pred={s:{'predicted_exponent':.5,'predicted_half_width':.01} for s in ('S1','S2','S3')}
    m=M.new_manifest('P5',False,'mock',cfg.__dict__,'mock');M.seal_predictions(m,pred,'test')
    fitted={s:{'headroom_ok':False} for s in pred}
    fitted['S1']={'headroom_ok':True,'fitted_exponent':.5,'fitted_se':.001,'n_replicates':4}
    got=p5.verdicts(m,{'identification':'IDENTIFIED'}, {'sealed_predictions':pred,'fitted':fitted},cfg)
    assert got['PREDICTION']=='INCONCLUSIVE' and got['assigned_systems']==3 and got['headroom_failures']==2


def test_monte_carlo_confidence_label_and_interval_follow_requested_level():
    from arc_instruments.mc_interval import RateWithUncertainty, clopper_pearson
    from scipy.stats import beta
    got=RateWithUncertainty(3,20,40,conf=.8)
    assert '80.0%' in got.render()
    assert clopper_pearson(3,20,.8)==pytest.approx((beta.ppf(.1,3,18),beta.ppf(.9,4,17)))
    assert got.as_dict()['wilson_two_sided'][1]<RateWithUncertainty(3,20,40,.95).as_dict()['wilson_two_sided'][1]
    with pytest.raises(ValueError): RateWithUncertainty(21,20,40)


def test_suite_replicates_restart_at_seed_and_keep_consecutive_rounds(tmp_path):
    store=CD.CheckpointStore(str(tmp_path/'store'));store.save('seed',CD.new_artefact(''))
    pool=CD.reference_pool(2)
    ladder=CD.SuiteLadder(pool,batch_runner=CD.inprocess_batch_runner())
    rounds=[]
    class Adapter:
        def revise(self,artefact,retained,task,rng):
            rounds.append(artefact['rounds']);out=dict(artefact);out['rounds']+=1;return out
    src=CD.suite_margin_source(Adapter(),ladder,store,'task',lambda *x:(1,1),8)
    rng=np.random.default_rng(0)
    for r in (0,1,0,1): src('dose+0.3',2.3,r,rng)
    assert rounds==[0,1,0,1]
    with pytest.raises(ValueError): src('dose+0.3',2.3,3,rng)


def test_p16_replay_reads_the_raw_series_not_just_stored_fits(tmp_path):
    run=demo()
    # Rewrite a raw trace while retaining the old fit and terminal summary, then reseal the local bundle.
    # Replay must expose the disagreement even though the bundle hashes are internally consistent.
    # The series is where the measurement lives, so that is what is rewritten: the arm keeps its
    # terminal summary and its stored fit, and the replay recomputes both from the rewritten series.
    run['arm_series'][0]['margin']=[-float(i) for i in range(30)]
    evidence.write_bundle(run,tmp_path/'run')
    with pytest.raises(ValueError,match='reproduce'): evidence.replay_bundle(tmp_path/'run')

    # And a bundle whose two halves disagree is refused rather than silently reconciled: a series
    # written onto the arm as well, contradicting the one beside it, is a record that cannot be read.
    other=demo()
    other['arms'][0]['margin']=[-float(i) for i in range(30)]
    evidence.write_bundle(other,tmp_path/'disagreeing')
    with pytest.raises(ValueError,match='disagree'):
        evidence.replay_bundle(tmp_path/'disagreeing')
