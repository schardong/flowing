landmark_models/shape_predictor_68_face_landmarks_GTX.dat:
	@echo "Downloading DLib GTX 68 landmarks detection model"
	@curl --location --remote-header-name --remote-name https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2
	@bunzip2 shape_predictor_68_face_landmarks_GTX.dat.bz2
	@mkdir -p landmark_models
	@mv shape_predictor_68_face_landmarks_GTX.dat landmark_models/
	@rm -Rf shape_predictor_68_face_landmarks_GTX.dat.bz2
	@echo "DLib GTX 68 landmarks detection model downloaded"

data/frll_neutral_front:
	@echo "Downloading the FRLL dataset"
	@curl --location --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" --user-agent "https://figshare.com/" --remote-header-name --remote-name https://figshare.com/ndownloader/files/8541961
	@unzip neutral_front.zip
	@mkdir -p $@
	@mv neutral_front/* $@
	@rm -Rf __MACOSX neutral_front.zip neutral_front/ data/frll_neutral_front/*.tem
	@echo "Dataset downloaded"

data/frll_neutral_front_cropped: data/frll_neutral_front landmark_models/shape_predictor_68_face_landmarks_GTX.dat
	@echo "Cropping face images"
	@python standalone/align.py --just-crop --output-size 1350 --n-tasks 4 $< $@
	@echo "Face images cropped"
