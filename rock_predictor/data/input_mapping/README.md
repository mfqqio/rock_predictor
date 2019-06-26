# Expected Data in `input_mapping` Directory

There are 3 .csv files to provide in the `input_mapping` folder:
* rock_class_mapping.csv
* explosive_by_rock_class.csv
* telemetry_mapping.csv

*Note: These files will need to be named the same as listed above.*

#### rock_class_mapping.csv
A mapping of rock type to rock class where rock types are sub-types of rocks that get grouped together by their rock class. Rock class is the target label. For example:

|rock_type|rock_class|
|---------|----------|
|HEMATITE1|HEMATITE|
|HEMATITE2|HEMATITE|
|HEMATITE3|HEMATITE|
|MAGNETITE_SPECIAL|MAGNETITE|

#### explosive_by_rock_class.csv
A mapping of rock class to the amount of explosive used in kg/m3 and kg/t. For example:

|rock_class |	kg/m3	| kg/t |
|-----------|-------|------|
|HEMATITIE|1.5|0.5|
|MAGNETITIE|1.0|0.4|

#### telemetry_mapping.csv
A mapping of a numerical number (FieldId from telemetry data) to the name of the actual sensor measurement collected by the drilling equipment.

|FieldId |FieldDesc  |
|--------|-----------|
|1000 |Rotation Speed|
|1002 |Head Vibration|
