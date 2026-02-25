import sqlite3
import sys

# Rocky Linux and other RHEL-based distros ship with SQLite < 3.35.0,
# which is too old for ChromaDB. pysqlite3-binary bundles a newer SQLite.
if sqlite3.sqlite_version_info < (3, 35, 0):
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
