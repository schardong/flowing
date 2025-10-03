landmark_models/shape_predictor_68_face_landmarks_GTX.dat:
	@echo "Downloading DLib GTX 68 landmarks detection model"
	@curl --location --remote-header-name --remote-name https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2
	@bunzip2 shape_predictor_68_face_landmarks_GTX.dat.bz2
	@mkdir -p landmark_models
	@mv shape_predictor_68_face_landmarks_GTX.dat landmark_models/
	@rm -Rf shape_predictor_68_face_landmarks_GTX.dat.bz2
	@echo "Done"

data/frll_neutral_front:
	@echo "Downloading the FRLL dataset"
	@curl --location --remote-header-name --remote-name https://figshare.com/ndownloader/files/8541961
	@unzip neutral_front.zip
	@mkdir -p $@
	@mv neutral_front/* $@
	@rm -Rf __MACOSX neutral_front.zip neutral_front/
	@echo "Dataset downloaded"

data/frll_neutral_front_cropped: data/frll_neutral_front landmark_models/shape_predictor_68_face_landmarks_GTX.dat
	@python standalone/align.py --just-crop --output-size 1350 --n-tasks 4 $< $@
