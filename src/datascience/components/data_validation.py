from src.datascience.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_cols_status = True
            validation_datatype_status=True
            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
            schema=self.config.all_schema
            
            for col in all_cols:
                if col not in all_schema:
                    validation_cols_status = False
                    break
                if str(data[col].dtype)!=str(schema[col]):
                    validation_datatype_status=False
                    break

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f'last validation column: {col}\n')
                f.write(f"Validation columns status: {validation_cols_status}\n")
                f.write(f"Validation datatype status: {validation_datatype_status}")
            
            return validation_cols_status and validation_datatype_status
        
        except Exception as e:
            raise e

    