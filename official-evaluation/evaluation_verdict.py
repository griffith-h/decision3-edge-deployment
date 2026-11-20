"""Final-release gate follows the explicitly requested evaluation scope."""
from stages import PIN


def require_final_complete(verdict):
    if verdict.get('final_stage') != 'native-bf16' or verdict.get('model_revision') != PIN:
        raise ValueError('Wrong final stage or model revision')
    final = next((r for r in verdict.get('stages', []) if r['stage'] == 'native-bf16'), None)
    expected = verdict.get('requests', 0)
    if not (verdict.get('final_stage_evaluation_complete') and final and expected > 0
            and final.get('sample_complete') and final.get('completed_requests') == expected
            and final.get('expected_requests') == expected and 'board_method' in final
            and sum(final.get('counts', {}).values()) == expected
            and verdict.get('sample_rows_sha256')):
        raise ValueError('Final version must finish the entire frozen sample before release')
    return final
