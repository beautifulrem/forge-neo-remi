"""The fixed sentences a job or a model load can end with.

They are stored and sent as they are (English). The SPA translates them where it
shows them (the history, the failure toasts, the pipeline runs), so each one is a key
of the server catalogs: ``tests/remi_core_strings.py`` collects :data:`JOB_TEXTS` and
:data:`CHAIN_TEXTS`. A job that
failed in its own code ends with the exception's text instead, a diagnostic that
is shown as it is."""

SERVER_STOPPED = "the server stopped while the job ran"
OUT_OF_MEMORY = "Out of memory: the model was unloaded"
END_NOT_SAVED = "the job's end could not be saved"
DELETED = "deleted from the queue"
NOT_REPLAYABLE = "not replayable: its arguments do not map to the current parameters"
MODEL_NOT_LOADED = "the model was not loaded"

JOB_TEXTS = (SERVER_STOPPED, OUT_OF_MEMORY, END_NOT_SAVED, DELETED, NOT_REPLAYABLE, MODEL_NOT_LOADED)

# The fixed sentences a pipeline (and the stage that ended it) can end with; a stage job's own
# failure ends it with that job's text instead.
STAGE_SERVER_STOPPED = "the server stopped while the stage ran"
STAGE_JOB_DELETED = "the stage job was deleted"
STAGE_NO_IMAGE = "the stage made no image to pass on"
STAGE_JOB_FAILED = "stage job failed"
STAGE_JOB_STOPPED = "the stage job was stopped"
UPSAMPLE_NOTHING = "prompt upsampling returned nothing"

CHAIN_TEXTS = (STAGE_SERVER_STOPPED, STAGE_JOB_DELETED, STAGE_NO_IMAGE, STAGE_JOB_FAILED, STAGE_JOB_STOPPED, UPSAMPLE_NOTHING)
