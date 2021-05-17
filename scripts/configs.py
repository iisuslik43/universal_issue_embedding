from scripts.dataset.creating.config import DatasetCreatingConfig
from scripts.dataset.preprocessing.config import PreprocessingConfig, DUPLICATE_TARGET

CUSTOM_FIELDS = {
    'Rider': ['Type', 'Assignee', 'State', 'Subsystem', 'User priority', 'Technology', 'OS'],
    'Kotlin': ['Type', 'Assignee', 'State', 'Subsystems', 'Priority'],
    'IDEA': ['Type', 'Assignee', 'State', 'Subsystem', 'Priority', 'Tester']
}

CONFIGS = {
    'Rider': {
        'creating': DatasetCreatingConfig(
            custom_fields=CUSTOM_FIELDS['Rider']
        ),
        'Subsystem': {
            'preprocessing': PreprocessingConfig(
                target_field='Subsystem',
                custom_fields=CUSTOM_FIELDS['Rider'],
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split(' - ')[0]
                }
            )
        },
        'Duplicate': {
            'preprocessing': PreprocessingConfig(
                target_field=DUPLICATE_TARGET,
                custom_fields=CUSTOM_FIELDS['Rider'],
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split(' - ')[0]
                }
            )
        },
        'Priority': {
            'preprocessing': PreprocessingConfig(
                target_field='User priority',
                custom_fields=CUSTOM_FIELDS['Rider'],
                percent_for_other=0.05,
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split(' - ')[0],
                    'User priority': lambda x: 'NORMAL' if x in ['Normal', 'Nice to have'] else 'HIGH'
                }
            )
        }
    },
    'Kotlin': {
        'creating': DatasetCreatingConfig(
            custom_fields=CUSTOM_FIELDS['Kotlin']
        ),
        'Subsystem': {
            'preprocessing': PreprocessingConfig(
                target_field='Subsystems',
                custom_fields=CUSTOM_FIELDS['Kotlin'],
                custom_fields_mappings={
                    'Subsystems': lambda x: x.split('.')[0],
                }
            )
        },
        'Duplicate': {
            'preprocessing': PreprocessingConfig(
                target_field=DUPLICATE_TARGET,
                custom_fields=CUSTOM_FIELDS['Kotlin'],
                custom_fields_mappings={
                    'Subsystems': lambda x: x.split('.')[0]
                }
            )
        },
        'Priority': {
            'preprocessing': PreprocessingConfig(
                target_field='Priority',
                custom_fields=CUSTOM_FIELDS['Kotlin'],
                custom_fields_mappings={
                    'Subsystems': lambda x: x.split('.')[0],
                    'Priority': lambda x: 'NORMAL' if x in ['Normal', 'Minor'] else 'HIGH'
                }
            )
        }
    },
    'IDEA': {
        'creating': DatasetCreatingConfig(
            custom_fields=CUSTOM_FIELDS['IDEA']
        ),
        'Subsystem': {
            'preprocessing': PreprocessingConfig(
                target_field='Subsystem',
                custom_fields=CUSTOM_FIELDS['IDEA'],
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split('.')[0],
                }
            )
        },
        'Duplicate': {
            'preprocessing': PreprocessingConfig(
                target_field=DUPLICATE_TARGET,
                custom_fields=CUSTOM_FIELDS['IDEA'],
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split('.')[0]
                }
            )
        },
        'Priority': {
            'preprocessing': PreprocessingConfig(
                target_field='Priority',
                custom_fields=CUSTOM_FIELDS['IDEA'],
                custom_fields_mappings={
                    'Subsystem': lambda x: x.split('.')[0],
                    'Priority': lambda x: 'NORMAL' if x in ['Normal', 'Minor'] else 'HIGH'
                }
            )
        }
    }
}
