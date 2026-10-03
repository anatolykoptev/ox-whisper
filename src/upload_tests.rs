//! Uploads must never strand a file: the temp dir is a tmpfs charged to the
//! container's memory limit, and a full one fails every request while
//! `/health` stays green.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, State};
use axum::http::Request;
use tokio_stream::wrappers::ReceiverStream;

use super::*;
use crate::config::Config;
use crate::handlers::AppState;
use crate::models::Models;
use crate::tmpfile::{listing, scratch_dir};

const BOUNDARY: &str = "oxw-test-boundary";

fn file_part(name: &str, data: &[u8]) -> Vec<u8> {
    let mut part = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"a.ogg\"\r\n\
         Content-Type: audio/ogg\r\n\r\n"
    )
    .into_bytes();
    part.extend_from_slice(data);
    part.extend_from_slice(b"\r\n");
    part
}

fn text_field(name: &str, value: &str) -> Vec<u8> {
    format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
        .into_bytes()
}

fn closing() -> Vec<u8> {
    format!("--{BOUNDARY}--\r\n").into_bytes()
}

fn request(body: Body) -> Request<Body> {
    Request::builder()
        .method("POST")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(body)
        .unwrap()
}

async fn multipart_of(body: Vec<u8>) -> Multipart {
    Multipart::from_request(request(Body::from(body)), &())
        .await
        .unwrap()
}

/// A body that delivers `sent`, then never ends: a client that stalls
/// mid-upload. Dropping the sender is what ends it, so the returned sender
/// must be kept alive for the stall.
fn stalled_body(
    sent: Vec<u8>,
) -> (
    Body,
    tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    tx.try_send(Ok(Bytes::from(sent))).unwrap();
    (Body::from_stream(ReceiverStream::new(rx)), tx)
}

fn state_with_upload_dir(dir: &std::path::Path) -> Arc<AppState> {
    let mut config = Config::from_lookup(&|_| None);
    config.upload_dir = dir.to_path_buf();
    Arc::new(AppState {
        models: Models::empty(),
        config,
        shutdown: crate::server::ShutdownSignal::inert(),
    })
}

#[tokio::test]
async fn a_second_audio_part_is_refused_and_the_first_is_not_stranded() {
    let dir = scratch_dir("upload-two");
    let mut body = file_part("file", b"first");
    body.extend(file_part("file", b"second"));
    body.extend(closing());

    let err = match parse_openai_upload(&mut multipart_of(body).await, &dir).await {
        Err(e) => e,
        Ok(_) => panic!("two audio parts must be refused"),
    };
    assert!(err.contains("only one audio file"), "{err}");
    assert!(listing(&dir).is_empty(), "stranded: {:?}", listing(&dir));
    std::fs::remove_dir(&dir).unwrap();
}

/// Positive control for the tests around it: the happy path does write the
/// file, owns it while the upload lives, and removes it when the upload goes.
#[tokio::test]
async fn an_upload_owns_its_file_until_dropped() {
    let dir = scratch_dir("upload-ok");
    let mut body = file_part("file", b"audio-bytes");
    body.extend(text_field("language", " RU "));
    body.extend(closing());

    let upload = parse_openai_upload(&mut multipart_of(body).await, &dir)
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(crate::language::resolve(&upload.language), Ok(Some("ru")));
    assert_eq!(listing(&dir).len(), 1);
    assert_eq!(std::fs::read(upload.file.path()).unwrap(), b"audio-bytes");
    drop(upload);
    assert!(listing(&dir).is_empty());
    std::fs::remove_dir(&dir).unwrap();
}

/// A body that ends in the middle of a later part used to read as "no more
/// fields": the file was transcribed with the fields after it silently lost
/// (a `language` that never arrived).
#[tokio::test]
async fn a_truncated_body_is_an_error_not_a_shorter_request() {
    let dir = scratch_dir("upload-truncated");
    // The file and one field arrive intact; the next part's header never
    // finishes.
    let mut body = file_part("file", b"audio");
    body.extend(text_field("model", "whisper-1"));
    body.extend(b"--oxw-test-boundary\r\nContent-Disposition: form-data; name=\"lang".to_vec());

    let res = parse_openai_upload(&mut multipart_of(body).await, &dir).await;
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("a truncated body must not parse"),
    };
    assert!(err.contains("multipart"), "{err}");
    assert!(listing(&dir).is_empty(), "stranded: {:?}", listing(&dir));
    std::fs::remove_dir(&dir).unwrap();
}

/// The real handler, dropped mid-request the way axum drops it when the client
/// times out: the upload was already written, and must be gone afterwards.
#[tokio::test]
async fn a_handler_dropped_mid_request_leaves_no_file() {
    let dir = scratch_dir("upload-cancel");
    let state = state_with_upload_dir(&dir);

    // The file part is complete (and written); the next part's header never
    // finishes, so the handler is parked reading the body.
    let mut sent = file_part("file", &[7u8; 4096]);
    sent.extend(b"--oxw-test-boundary\r\nContent-Disposition: form-data; name=\"langu".to_vec());
    let (body, _keep_stalled) = stalled_body(sent);
    let multipart = Multipart::from_request(request(body), &()).await.unwrap();

    let handler = tokio::spawn(crate::handler_openai::transcriptions(
        State(state),
        multipart,
    ));

    // It must reach the stall with the file on disk, or the check below proves
    // nothing.
    let mut waited = Duration::ZERO;
    while listing(&dir).is_empty() {
        assert!(waited < Duration::from_secs(5), "upload was never written");
        tokio::time::sleep(Duration::from_millis(10)).await;
        waited += Duration::from_millis(10);
    }
    assert!(!handler.is_finished(), "handler must still be waiting");

    handler.abort();
    assert!(handler.await.unwrap_err().is_cancelled());
    assert!(
        listing(&dir).is_empty(),
        "file stranded by a dropped handler: {:?}",
        listing(&dir)
    );
    std::fs::remove_dir(&dir).unwrap();
}

/// The handler is dropped while its blocking decode job is running: the job
/// must still find its input, and the file must be gone once the job ends. A
/// job holding only a path (the guard staying with the handler) would read a
/// file that vanished the moment the handler was dropped.
#[tokio::test]
async fn a_file_is_not_deleted_under_a_running_job() {
    let dir = scratch_dir("upload-job");
    let mut config = Config::from_lookup(&|_| None);
    config.upload_dir = dir.clone();
    config.decode_delay = Duration::from_millis(600);
    let state = Arc::new(AppState {
        models: Models::empty(),
        config,
        shutdown: crate::server::ShutdownSignal::inert(),
    });

    let mut body = file_part("file", &[3u8; 2048]);
    body.extend(closing());
    let handler = tokio::spawn(crate::handler_openai::transcriptions(
        State(state),
        multipart_of(body).await,
    ));

    let events = |stage: &str| -> Vec<bool> {
        crate::transcribe::probe::EVENTS
            .lock()
            .unwrap()
            .iter()
            .filter(|(p, s, _)| p.starts_with(&dir) && *s == stage)
            .map(|(_, _, exists)| *exists)
            .collect()
    };
    let wait_for = |stage: &'static str| {
        let events = &events;
        async move {
            for _ in 0..500 {
                if !events(stage).is_empty() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            panic!("job never reached {stage}");
        }
    };

    wait_for("started").await; // the job is running, stalled
    handler.abort();
    assert!(handler.await.unwrap_err().is_cancelled());

    wait_for("after_stall").await;
    assert_eq!(
        events("after_stall"),
        vec![true],
        "the input vanished under a running job"
    );
    for _ in 0..500 {
        if listing(&dir).is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        listing(&dir).is_empty(),
        "stranded after the job: {:?}",
        listing(&dir)
    );
    std::fs::remove_dir(&dir).unwrap();
}
