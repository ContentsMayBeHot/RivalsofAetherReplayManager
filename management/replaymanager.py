import configparser
import enum
import os
import random
import shutil

import skimage.exposure
import numpy as np

import utilites as utls


class ReplayFile:
    '''Wrapper class used for handling individual replay files as well as their
        frame and label dumps.

    # Arguments
        replay_path: Path to the replay file.
        frames_root_path: Root folder for frame dumps. This replay will dump its
            frames to frames_path/id/.
        labels_root_path: Root folder for label dumps. This replay will dump its
            labels to labels_path/id/.

    # Notes
        The game generates unique file names for each replay, so they are used
            as IDs.
    '''

    def __init__(self, replay_path, frames_root_path=None,
            labels_root_path=None):
        self.__filename = os.path.basename(replay_path)
        self.__filepath = os.path.abspath(replay_path)
        if frames_root_path:
            self.__frames_path = os.path.join(frames_root_path, self.__filename)
        if labels_root_path:
            self.__labels_path = os.path.join(labels_root_path, self.__filename)

    @property
    def name(self):
        return self.__filename

    @property
    def path(self):
        return self.__filepath

    @property
    def frames_path(self):
        return self.__frames_path

    @property
    def labels_path(self):
        return self.__labels_path

    def set_frames_path(self, frames_path):
        self.__frames_path = frames_path

    def set_labels_path(self, labels_path):
        self.__labels_path = labels_path

    def get_version(self, simple=False):
        '''Read the file and retrieve the string representing which version of
            the game it is compatible with.

        # Arguments
            (optional) simple: Reformat the version string so that it reads
                x.y.z rather than xx_yy_zz.

        # Returns
            String representing the game version.
        '''
        with open(self.__filepath, 'r') as f:
            line = f.readline()
            version = '{}_{}_{}'.format(
                    str(line[1:3]), str(line[3:5]), str(line[5:7]))
        if simple:
            return utls.dname_to_version(version)
        return line

    def save_frame_and_labels(self, frame, frame_id, labels):
        self.save_frame(frame, frame_id)
        self.save_labels(labels)

    def save_frame(self, frame, frame_id):
        '''Save a single game frame to the file system as a pickled NumPy array.

        # Arguments
            frame: NumPy matrix containing frame buffer.
            frame_id: An integer. Typically the offset from frame 0. For
                example, the 100th frame would have a frame_id value of 99.

        # Returns
            Path to the saved frame.

        # Notes
            This will save a pickled NumPy array to
                frames/replay_id/frame_id.npy
        '''
        frame_filename = str(frame_id) + '.npy'
        frame_filepath = os.path.join(self.__frames_path, frame_filename)
        with open(frame_filepath, 'wb') as fout:
            np.save(fout, frame)
        return frame_filepath

    def save_labels(self, labels):
        '''Save a set of player action matrices as a pickled NumPy array.
        
        # Arguments
            labels: A NumPy matrix containing action states for each player.

        # Returns
            List of paths to the saved labels.

        # Notes
            This will save from 1 to 4 pickled NumPy arrays to
                labels/replay_id/player_id.npy
        '''
        results = []
        for i, actions_matrix in enumerate(labels):
            label_filename = 'roa_' + str(i) + '.npy'
            label_filepath = os.path.join(self.__labels_path, label_filename)
            with open(label_filepath, 'wb') as fout:
                np.save(fout, np.array(actions_matrix, dtype=object))
            results.append(label_filepath)
        return results


class ReplayCollection:
    '''Represents a collection of replay files that occupy the same folder and
        are compatible with the same game version.

    # Arguments
        collection_path: Path to the folder containing the replay files.
        frames_root_path: Path to the root frames folder under which each 
            replay's frame data should be dumped.
        labels_root_path: Path to the root labels folder under which each 
            replay's label data should be dumped.

    # Notes
        Replays are considered "visited" if frame and label data exists for them
            or if they are popped from the unvisited list.
    '''
    
    def __init__(self, collection_path, frames_root_path, labels_root_path):
        self.__name = os.path.dirname(collection_path)
        self.__collection_path = os.path.abspath(collection_path)
        self.__frames_root_path = os.path.abspath(frames_root_path)
        self.__labels_root_path = os.path.abspath(labels_root_path)
        self.__replays = [
            ReplayFile(replay_path) for replay_path in
            utls.listdir_replays_only(self.__collection_path, as_path=True)
            ]
        self.__replays_unvisited = []
        self.reset_unvisited()

    @property
    def name(self):
        return self.__name

    @property
    def path(self):
        return self.__collection_path

    @property
    def frames_root_path(self):
        return self.__frames_root_path

    @property
    def labels_root_path(self):
        return self.__labels_root_path

    @property
    def count(self):
        return len(self.__replays)

    @property
    def count_unvisited(self):
        return len(self.__replays_unvisited)

    def reset_unvisited(self):
        '''Restore the list of unvisited replays to its initial state.

        # Returns
            Updated count of unvisited replays.
        '''
        self.__replays_unvisited = [
            replay for replay in self.__replays
            if not self.is_replay_collected(replay)
            ]
        return len(self.__replays_unvisited)

    def is_replay_collected(self, replay):
        '''Test to see if frame and label data exists for a given replay.

        # Arguments
            replay: Replay object to test.

        # Returns
            False if frame or label data does not exist, or else True.
        '''
        replay_frames_path = os.path.join(self.__frames_root_path, replay.name)
        replay_labels_path = os.path.join(self.__labels_root_path, replay.name)
        if not os.path.isdir(replay_frames_path):
            return False
        if not os.path.isdir(replay_labels_path):
            return False
        if not os.listdir(replay_frames_path):
            return False
        if not os.listdir(replay_labels_path):
            return False
        return True

    def pop_unvisited(self):
        '''Pop a replay from the front of unvisited list.

        # Returns
            The next replay object from the unvisted list, or else None if the
                list is empty.

        # Notes
            The replay returned by this method will be removed from the
                unvisited list. If frames or labels are not generated, then this
                removal can be undone by either calling reset_unvisited() or
                reinstantiating the ReplayCollection.
        '''
        if not self.__replays_unvisited:
            return None
        return  self.__replays_unvisited.pop(0)


class ReplayManager:
    '''Class for managing replays and replay collections.

    # Arguments
        (optional) replays_path: Path to the game's replays folder. If left
            unspecified, this will be derived from the game version noted in
            the roa.ini file.
        (optional) destination_path: Path to the root folder where frame
            and label dumps should be saved. If left unspecified, this will
            be the game's repalys folder.
        (optional) skip_folder_creation: Set this to true to prevent creation
            of folders for frames and labels.
        (optional) skip_backups: Set this to true to prevent copying of all
            replay files found in the game's replays folder to replays/backup.

    # Notes
        Instantiation of this class can result in the creation of several
            folders on the local file system. Also note that there is
            nothing preventing the concurrent use of multiple ReplayManager
            instances, but such usage is neither tested nor intended.
    '''

    def __init__(self, replays_path=None, destination_root_path=None,
            skip_folder_creation=False, skip_backups=False):
        # Establish path to the game's replays folder
        if not replays_path:
            self.__config = configparser.ConfigParser()
            config_path = os.path.join(
                    os.path.abspath(os.path.dirname(__file__)), 'roa.ini')
            self.__config.read(config_path)
            replays_path = self.__config['RivalsofAether']['PathToReplays']
        self.__replays_path = replays_path

        # Establish root path for frame and label dumps
        if not destination_root_path:
            destination_root_path = replays_path
        self.__destination_root_path = destination_root_path

        # Establish destination subdirectory paths
        self.__frames_root_path = os.path.join(
                self.__destination_root_path, 'frames')
        self.__labels_root_path = os.path.join(
                self.__destination_root_path, 'labels')

        if not skip_folder_creation:
            # Ensure target paths exist
            utls.ensure_directory_exists(self.__frames_root_path)
            utls.ensure_directory_exists(self.__labels_root_path)

        if not skip_backups:
            # Backup any replays in the replays folder
            self.__backup_replays__()

        # Declare ReplayCollection attribute
        self.__collection = None

    def load_collection(self, collection_path=None):
        '''Load a replay collection from a given path.

        # Arguments
            collection_path: Path to collection directory.

        # Returns
            String identifier for collection.

        # Notes
            This initializes an internal ReplayCollection, which is used to
                track a group of replay source files that 1) exist in the same
                directory, and 2) belong to the same game version.
        '''
        if not collection_path:
            # Derive collection path from version string
            game_version = self.__config['RivalsofAether']['GameVersion']
            game_version = utls.version_to_dname(game_version)
            collection_path =  os.path.join(self.__replays_path, game_version)
        # Instantiate collection
        self.__collection = ReplayCollection(
                collection_path, self.__frames_root_path, 
                self.__frames_root_path)
        return self.__collection.name

    def load_next_replay(self, skip_deletions=False):
        '''Load the next unvisited replay from the replay collection. The actual
            replay file will be copied into the game's replays folder.
        
        # Arguments
            (optional) skip_deletions: Set to true to prevent this method 
                from deleting any replay files from the game's replays folder.
                Skipping deletions may cause the Rivals of Aether replays menu
                to crash if an incompatible replay file is left behind.

        # Returns
            Replay object representing the next replay.

        # Notes
            A side effect of calling this method is that every replay file found
                in the game's replays folder will be deleted. By default,
                ReplayManager will make a backup to avoid accidental deletion.
        '''
        # Clear the replays folder
        self.__flush_replays__()

        # Get the next unvisited replay
        replay = self.__collection.pop_unvisited()

        # Copy the replay to the replays folder
        shutil.copy(replay.path, self.__replays_path)

        # Get frames and labels paths for the replay
        replay_frames_path = os.path.join(self.__frames_root_path, replay.name)
        replay_labels_path = os.path.join(self.__labels_root_path, replay.name)
        replay.set_destination_paths(replay_frames_path, replay_labels_path)

        # Ensure frames and labels paths exist
        utls.ensure_directory_exists(replay_frames_path)
        utls.ensure_directory_exists(replay_labels_path)

        # Return the replay object
        return replay

    def __backup_replays__(self):
        '''Copy all replays from the game's replays folder into a new folder
            located in replays/backup.
        '''
        backup_path = os.path.join(self.__replays_path, 'backup')
        utls.ensure_directory_exists(backup_path)
        for replay_path in utls.listdir_replays_only(
                self.__replays_path, as_path=True):
            os.rename(replay_path, backup_path)

    def __flush_replays__(self):
        '''Delete all replays from the game's replays folder.
        '''
        for replay_path in utls.listdir_replays_only(
                self.__replays_path, as_path=True):
            os.remove(replay_path)

    def get_collections(self):
        '''Build a dictionary in the following format:
            { game_version: [ replay_0, replay_1, ... replay_n ] }.

        # Returns
            Dictionary containing replays mapped to game version strings.
        '''
        collections = {}
        for replay_path in utls.listdir_replays_only(
                self.__replays_path, as_path=True):
            replay = ReplayFile(replay_path)
            game_version = replay.get_version()
            if not collections.get(game_version):
                collections[game_version] = []
            collections[game_version].append(replay)
        return collections
            
    def make_collections(self):
        '''Move each replay file from the game's replay folder into a folder
            representing that replay's game version.
        '''
        for replay_path in utls.listdir_replays_only(
                self.__replays_path, as_path=True):
            # Get the replay's game version
            replay = ReplayFile(replay_path)
            game_version = replay.get_version()

            # Get the path to the collection
            collection_path = os.path.join(
                    self.__destination_root_path, game_version)
            utls.ensure_directory_exists(collection_path)

            # Move the replay to the new directory
            replay_new_path = os.path.join(collection_path, replay.name)
            os.rename(replay_path, replay_new_path)

