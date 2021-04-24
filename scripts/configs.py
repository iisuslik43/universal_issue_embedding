from scripts.dataset.creating.config import DatasetCreatingConfig
from scripts.dataset.preprocessing.config import PreprocessingConfig

CUSTOM_FIELDS = {
    'Rider': ['Type', 'Assignee', 'State', 'Subsystem', 'User priority', 'Technology', 'OS']
}

CONFIGS = {
    'Rider': {
        'creating': DatasetCreatingConfig(
            custom_fields=CUSTOM_FIELDS['Rider']
        ),
        'preprocessing': PreprocessingConfig(
            target_field='Subsystem',
            custom_fields=CUSTOM_FIELDS['Rider'],
            custom_fields_mappings={
                'Subsystem': lambda x: x.split(' - ')[0],
                'User priority': lambda x: 'NORMAL' if x in ['Normal', 'Nice to have'] else 'HIGH'
            }
        )
    }
}
