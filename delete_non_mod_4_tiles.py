from timemachine import TimeMachine

# This script takes a list of directories, looks for .timemachine/tm.json as a timemachine marker,
# and deletes all the non-mod-4 tiles in each timemachine

# Usage:
# python delete_non_mod_4_tiles.py [--delete] <directory1> <directory2> ... <directoryN>

# If --delete is specified, the script will delete the non-mod-4 tiles
# If --delete is not specified, the script will just print the number of tiles in each directory that would be deleted

# Parse args

import argparse

parser = argparse.ArgumentParser(description='Delete non-mod-4 tiles from TimeMachine directories')
parser.add_argument('--delete', action='store_true', help='Actually delete the files (default: dry run)')
parser.add_argument('directories', nargs='+', help='Directories to process')

args = parser.parse_args()
delcount = 0

for directory in args.directories:
    timemachine_paths = TimeMachine.find_timemachine_paths_recursively(directory)
    print(f"{directory}: Found {len(timemachine_paths)} timemachine paths")
    for timemachine_path in sorted(timemachine_paths):
        tm = TimeMachine(timemachine_path, verbose=False)
        print(timemachine_path)
        to_delete = tm.delete_non_mod4_tiles(args.delete)
        delcount += len(to_delete)

if args.delete:
    print(f"Deleted {delcount} non-mod-4 tiles")
else:
    print(f"Would delete {delcount} non-mod-4 tiles")