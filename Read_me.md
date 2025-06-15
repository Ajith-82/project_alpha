STRATEGY 1
The 10-day SMA should be below the 30-day SMA. - 0
The MACD value should be above the MACD signal line. - 1
The MACD value should not be above 0. - 0
The 10-day and the 30-day SMA should be above the 50-day SMA. - 1
The 10-day, 30-day, and 50-day SMA should be below the 200-day SMA. - 0
STRATEGY 2
The 10-day SMA should be below the 30-day SMA.
The MACD value should be above the MACD signal line.
The MACD value should not be above 0.
The 10-day and the 30-day SMA should be above the 50-day SMA.
The 10-day, 30-day, and 50-day SMA should be above the 200-day SMA.
STRATEGY 3
The 10-day SMA should be above the 30-day SMA.
The MACD value should be above the MACD signal line.
The MACD value should not be above 0.
The 10-day and the 30-day SMA should be above the 50-day SMA.
The 10-day, 30-day, and 50-day SMA should be below the 200-day SMA.
STRATEGY 4
The 10-day SMA should be above the 30-day SMA.
The MACD value should be below the MACD signal line.
The MACD value should be above 0.
The 10-day and the 30-day SMA should be below the 50-day SMA.
The 10-day, 30-day, and 50-day SMA should be below the 200-day SMA.
STRATEGY 5
The 10-day SMA should be above the 30-day SMA.
The MACD value should be below the MACD signal line.
The MACD value should be above 0.
The 10-day and the 30-day SMA should be below the 50-day SMA.
The 10-day, 30-day, and 50-day SMA should be above the 200-day SMA.
## Database Caching

The application can store downloaded price data in a local SQLite database.  Using a database avoids repeatedly downloading the same history and speeds up subsequent runs.

Use the `--db-path` option to specify the location of the SQLite file when running `project_alpha.py`:

```bash
python src/project_alpha.py --db-path my_data.db
```

If a database is provided, new prices are fetched only for dates that are not already stored. All historical data will be read from the database instead of relying solely on pickle caches.

### Migrating old cache files

For existing pickle caches you can populate a new database with the helper script:

```bash
python scripts/migrate_pickle_to_db.py <pickle_file> <database>
```

This script reads the old cached dictionary and inserts its contents into the SQLite tables.

## Saving and Loading Model Parameters

Training the volatility model can be time consuming.  You can save the learned
parameters to disk and reuse them in later runs.

Use `--save-model <file>` to write the parameters after training and
`--load-model <file>` to initialize training from a previous run.  Providing
saved parameters allows you to resume training, reduces start-up time and
enables incremental learning on new data.
