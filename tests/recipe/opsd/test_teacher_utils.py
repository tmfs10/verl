from recipe.opsd.teacher_utils import select_sdpo_teacher_indices


def test_select_sdpo_teacher_indices_targets_only_failed_samples_by_default():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "b", "b"],
        correct=[0.0, 1.0, 1.0, 0.0],
    )

    assert teacher_indices == [1, None, None, 2]
    assert distill_active == [True, False, False, True]


def test_select_sdpo_teacher_indices_keeps_success_samples_inactive_without_peer_success():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "a"],
        correct=[1.0, 0.0, 0.0],
        distill_only_failed=False,
        exclude_self_success=True,
    )

    assert teacher_indices == [None, 0, 0]
    assert distill_active == [False, True, True]


def test_select_sdpo_teacher_indices_can_allow_self_teacher_for_single_success():
    teacher_indices, distill_active = select_sdpo_teacher_indices(
        uids=["a", "a", "a"],
        correct=[1.0, 0.0, 0.0],
        distill_only_failed=False,
        exclude_self_success=False,
    )

    assert teacher_indices == [0, 0, 0]
    assert distill_active == [True, True, True]
