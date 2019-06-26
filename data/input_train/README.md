# Expected Data in `input_train` Directory

There are 3 folders inside the `input_train` directory. Each folder requires a certain type of .csv file.

#### labels
Inside the `labels` folder, a .csv file containing the target labels is required. It must have the following columns:

|hole_id|x|y|z|LITHO|COLLAR_TYPE|PLANNED_RTYPE|BLAST|HOLE_NAME|
|-|-|-|-|-|-|-|-|-|
|Hole identifier|x coordinate of hole|y coordinate of hole|z coordinate of hole|Actual rock class label|String signifying type of hole|Baseline model rock class label|Blast identifier|Name of hole|

For example:

|hole_id|x|y|z|LITHO|COLLAR_TYPE|PLANNED_RTYPE|BLAST|HOLE_NAME|
|-|-|-|-|-|-|-|-|-|
|H-120-34-1567|14098.1208|12782.1234|78.0505|HEMATITE|ACTUAL|HEMATITE|146|1567|

#### production
Inside the `production` folder, a .csv file containing information about each drilled hole is required. It must have the following columns:

|DrillPattern|DesignX|DesignY|DesignZ|DesignDepth|ActualX|ActualY|ActualZ|ActualDepth|ColletZ|HoleID|FullName|FirstName|UTCStartTime|UTCEndTime|StartTimeStamp|EndTimeStamp|DrillTime|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Identifier for drill pattern|Design x coordinate|Design y coordinate|Design depth|Design depth|Actual x coordinate|Actual y coordinate|Actual z coordinate|Actual depth|NULL if hole is re-drilled, else depth + design Z|Hole identifier|Equipment identifier|Name of drill operator|UTC drill start time|UTC drill end time|Local drill start time|Local drill end time|Total drill time in seconds|

For example:

|DrillPattern|DesignX|DesignY|DesignZ|DesignDepth|ActualX|ActualY|ActualZ|ActualDepth|ColletZ|HoleID|FullName|FirstName|UTCStartTime|UTCEndTime|StartTimeStamp|EndTimeStamp|DrillTime|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|H-120-34-1567|14098.1208|12782.1234|78.0505|15.703016|615350.5644|5855238.256|676.899|15.70|0000|1567|42|Joe Smith|2019-05-23 14:51:18.417|2019-05-23 10:51:18.417|2019-05-23 15:13:31.823|2019-05-23 11:13:31.823|1333.0|

#### telemetry
Inside the `telemetry` folder, a .csv file containing the telemetry data collected by the drill is required. It must have the following columns:

|FieldTimestamp|FieldId|FieldData|FieldX|FieldY|
|-|-|-|-|-|
|UNIX timestamp when measurement was collected|Code for sensor parameter per `telemetry_mapping.csv` file in `input_mapping` folder|Measurement value|x coordinate of measurement|y coordinate eof measurement|

For example:

|FieldTimestamp|FieldId|FieldData|FieldX|FieldY|
|-|-|-|-|-|
|1561016962|1000|35.2|14098.1208|12782.1234|
