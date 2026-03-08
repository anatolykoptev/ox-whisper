use std::sync::Mutex;

pub struct Pool<T> {
    items: Mutex<Vec<T>>,
}

impl<T> Pool<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items: Mutex::new(items),
        }
    }

    /// Acquire an item from the pool. Returns None if pool is empty.
    pub fn acquire(&self) -> Option<PoolGuard<'_, T>> {
        let mut items = self.items.lock().ok()?;
        let item = items.pop()?;
        Some(PoolGuard {
            pool: self,
            item: Some(item),
        })
    }

    fn return_item(&self, item: T) {
        if let Ok(mut items) = self.items.lock() {
            items.push(item);
        }
    }
}

pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> std::ops::Deref for PoolGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item.as_mut().unwrap()
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.pool.return_item(item);
        }
    }
}
