import time

import pytest
import torch

from run_expanded_canonical_training import Guard, schedule, sequence_mse


def records():
    return [dict(motion_id=str(i), split='train', source_group_id=f'train-{i}', training_admissible=True)
            for i in range(4)] + [dict(motion_id='heldout', split='val', source_group_id='validation', training_admissible=False)]


def test_epoch_population_and_train_only_roles():
    rows = schedule(records(), epochs=3, batch_size=4, seed=7)
    assert len(rows) == 24
    for epoch in range(1, 4):
        assert {(r['motion_id'], r['sensor_config']) for r in rows if r['epoch'] == epoch} == {
            (str(i), c) for i in range(4) for c in ('head', 'wrists')}
    assert all(r['motion_id'] != 'heldout' for r in rows)
    assert rows == schedule(records(), 3, 4, 7)
    assert rows != schedule(records(), 3, 4, 8)


def test_reject_source_overlap_and_training_unadmitted():
    population = records(); population[-1]['source_group_id'] = 'train-0'
    with pytest.raises(ValueError, match='source overlap'):
        schedule(population, 3, 4, 7)
    population = records(); population[0]['training_admissible'] = False
    with pytest.raises(ValueError, match='training-admitted'):
        schedule(population, 3, 4, 7)


def test_no_unequal_accumulation_group():
    with pytest.raises(ValueError, match='Equal accumulation'):
        schedule(records(), 3, 3, 7)


def test_equal_sequence_weights_and_padding_has_no_gradient():
    prediction = torch.tensor([[[1.], [10000.], [10000.]], [[3.], [3.], [3.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss = sequence_mse(prediction, target, mask)
    assert loss.item() == 5.  # (one short sequence MSE=1 + long sequence MSE=9)/2
    loss.backward()
    assert torch.equal(prediction.grad[0, 1:], torch.zeros(2, 1))
    assert prediction.grad[0, 0, 0] == 1.
    assert torch.equal(prediction.grad[1], torch.ones(3, 1))


def test_deadline_and_budget_stop_without_cuda_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    guard = Guard(dict(status_path=str(tmp_path / 'status.json'), deadline_monotonic=time.monotonic() + 10,
                       memory_bytes=1))
    guard.count('mdm_calls', 1)
    with pytest.raises(RuntimeError, match='budget exceeded'):
        guard.count('mdm_calls', 1)
    guard.context['deadline_monotonic'] = time.monotonic() - 1
    with pytest.raises(TimeoutError, match='whole-attempt deadline'):
        guard.check()
