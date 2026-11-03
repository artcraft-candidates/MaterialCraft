// SPDX-License-Identifier: MIT OR Apache-2.0
//! A small persistent worker pool for the per-pixel loops.
//!
//! Starting operating-system threads for every node costs more than a small image takes to compute (the editor's quick
//! 256 px pass), so the workers are started once and reused. Work is handed out in small numbered pieces from a shared
//! counter, so a slow region of the image does not leave the other cores idle.
//!
//! Only one job uses the workers at a time. A job started while they are busy (another thread's job, or a loop nested
//! inside a running job) runs on short-lived scoped threads instead, as every loop did before the pool existed.
use std::cell::Cell;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Number of threads the loops use (cores available to the process), looked up once.
pub fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        std::env::var("MC_THREADS").ok().and_then(|s| s.parse::<usize>().ok()).filter(|v| *v > 0).unwrap_or(n).clamp(1, 256)
    })
}

/// One running job: `f(i)` for every `i` in `0..n`, taken from `next`.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
    failed: AtomicBool,
}

impl Job {
    /// Takes pieces until none are left. A panicking piece is recorded and the rest still run.
    fn work(&self) {
        // SAFETY: the job (and the closure it points to) lives until every thread has left it; see `for_each`.
        let f = unsafe { &*self.f };
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                break;
            }
            if self.failed.load(Ordering::Relaxed) {
                continue;
            }
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| f(i))) {
                self.failed.store(true, Ordering::Relaxed);
                if let Ok(mut slot) = self.panic.lock() {
                    slot.get_or_insert(p);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct JobPtr(*const Job);
// SAFETY: the pointer is only dereferenced while the job is alive (the owner waits for all workers to leave it).
unsafe impl Send for JobPtr {}

struct State {
    generation: u64,
    job: Option<JobPtr>,
    /// workers currently working on `job`
    inside: usize,
}

struct Pool {
    state: Mutex<State>,
    wake: Condvar,
    left: Condvar,
    /// held by the thread whose job the workers are running
    owner: Mutex<()>,
}

thread_local! {
    /// set while this thread runs pieces of a job (as owner or worker)
    static IN_JOB: Cell<bool> = const { Cell::new(false) };
}

fn pool() -> &'static Pool {
    static P: OnceLock<&'static Pool> = OnceLock::new();
    P.get_or_init(|| {
        let p: &'static Pool = Box::leak(Box::new(Pool { state: Mutex::new(State { generation: 0, job: None, inside: 0 }), wake: Condvar::new(), left: Condvar::new(), owner: Mutex::new(()) }));
        for k in 1..threads() {
            let started = std::thread::Builder::new().name(format!("mc-worker-{k}")).spawn(move || worker(p));
            if started.is_err() {
                break; // fewer helpers: the owner still does all the work that is left
            }
        }
        p
    })
}

fn worker(p: &'static Pool) {
    IN_JOB.with(|c| c.set(true));
    let mut seen = 0u64;
    loop {
        let job = {
            let mut st = match p.state.lock() {
                Ok(s) => s,
                Err(_) => return,
            };
            while st.generation == seen {
                st = match p.wake.wait(st) {
                    Ok(s) => s,
                    Err(_) => return,
                };
            }
            seen = st.generation;
            match st.job {
                Some(j) => {
                    st.inside += 1;
                    j
                }
                None => continue,
            }
        };
        // SAFETY: `inside` was raised under the lock while the job was published, so its owner waits for us.
        unsafe { &*job.0 }.work();
        if let Ok(mut st) = p.state.lock() {
            st.inside -= 1;
            if st.inside == 0 {
                p.left.notify_all();
            }
        }
    }
}

/// Withdraws the job and waits until no worker is inside it any more (also when the owner's own piece panicked).
struct Withdraw<'a>(&'a Pool);

impl Drop for Withdraw<'_> {
    fn drop(&mut self) {
        if let Ok(mut st) = self.0.state.lock() {
            st.job = None;
            while st.inside > 0 {
                st = match self.0.left.wait(st) {
                    Ok(s) => s,
                    Err(_) => return,
                };
            }
        }
    }
}

/// Runs `f(i)` for every `i` in `0..n` on all cores and returns when every call has finished. Calls may run in any order
/// and on any thread. A panic in `f` is passed on to the caller once all calls are done.
pub fn for_each(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n == 0 {
        return;
    }
    if n == 1 || threads() == 1 {
        (0..n).for_each(f);
        return;
    }
    if IN_JOB.with(|c| c.get()) {
        // nested inside a job (the workers are taken): short-lived threads keep this level parallel too
        return scoped(n, f);
    }
    let p = pool();
    let Ok(_own) = p.owner.try_lock() else {
        // the workers are busy with another thread's job: use short-lived threads for this one
        return scoped(n, f);
    };
    let fp: *const (dyn Fn(usize) + Sync + '_) = f;
    // SAFETY: the closure's lifetime is erased for the workers; `Withdraw` below keeps this frame (and so `f` and `job`)
    // alive until every worker has left the job, and no worker touches it after that.
    let fp: *const (dyn Fn(usize) + Sync + 'static) = unsafe { std::mem::transmute(fp) };
    let job = Job { f: fp, n, next: AtomicUsize::new(0), panic: Mutex::new(None), failed: AtomicBool::new(false) };
    let job_ptr = JobPtr(&job as *const Job);
    {
        let Ok(mut st) = p.state.lock() else { return scoped(n, f) };
        st.generation = st.generation.wrapping_add(1);
        st.job = Some(job_ptr);
    }
    p.wake.notify_all();
    {
        let _w = Withdraw(p);
        IN_JOB.with(|c| c.set(true));
        job.work();
        IN_JOB.with(|c| c.set(false));
    }
    if let Some(e) = job.panic.lock().ok().and_then(|mut s| s.take()) {
        resume_unwind(e);
    }
}

fn scoped(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let next = AtomicUsize::new(0);
    let run = || {
        IN_JOB.with(|c| c.set(true));
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            f(i);
        }
    };
    std::thread::scope(|s| {
        for _ in 1..threads().min(n) {
            s.spawn(run);
        }
        run();
    });
    IN_JOB.with(|c| c.set(false));
}

/// Splits `data` into pieces of `len` items and runs `f(piece_index, piece)` on all cores.
pub fn for_chunks<T: Send>(data: &mut [T], len: usize, f: &(dyn Fn(usize, &mut [T]) + Sync)) {
    let len = len.max(1);
    let pieces: Vec<Mutex<&mut [T]>> = data.chunks_mut(len).map(Mutex::new).collect();
    for_each(pieces.len(), &|i| {
        // every piece index is handed out once, so the lock is never contended
        if let Ok(mut c) = pieces[i].lock() {
            f(i, &mut c);
        }
    });
}

/// Rows per piece for an image `w` wide and `h` high: several pieces per core (so uneven rows balance out), but at
/// least a few thousand pixels each (so handing them out costs nothing).
pub fn rows_per_piece(w: usize, h: usize) -> usize {
    let by_cores = h.div_ceil(threads() * 6);
    let by_size = 4096usize.div_ceil(w.max(1));
    by_cores.max(by_size).clamp(1, h.max(1))
}
