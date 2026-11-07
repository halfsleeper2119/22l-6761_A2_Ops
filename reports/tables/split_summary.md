# Cleaning and splits

| step | images |
|---|---|
| files in audit | 975 |
| after dropping unreadable | 975 |
| after dropping cross-class duplicates | 975 |
| after keeping one copy per duplicate group | 974 |
| after dropping images smaller than 32px | 974 |

| split | total | real | ai | ai share |
|---|---|---|---|---|
| train | 681.0 | 305.0 | 376.0 | 0.552 |
| val | 146.0 | 65.0 | 81.0 | 0.555 |
| test | 147.0 | 66.0 | 81.0 | 0.551 |
