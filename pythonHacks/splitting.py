def train_test_split(X, y, test_ratio, seed=0):
    """Naključno razdeli podatke. Vrne (X_train, X_test, y_train, y_test)."""
    
    n = len(X)
    indices = list(range(n))
    
    # 2. Premešanje
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # 3. Izračun meje
    n_test = int(n * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 4. Ročna gradnja novih seznamov (brez fancy indexing-a)
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def train_test_split(X, y, test_ratio, seed=0):
    """Naključno razdeli podatke. Vrne (X_train, X_test, y_train, y_test)."""
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    n = len(X)
    indices = np.arange(n, dtype=int) # Zagotovi, da so indeksi int
    
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    n_test = int(n * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 2. Uporaba fancy indexinga, ki sedaj MORA delovati, ker sta X in y numpy polji
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def train_test_split(X, y, test_ratio, seed=0):
    # ... (tvoja dosedanja logika z indeksi) ...
    
    # 4. Ročna gradnja seznamov (kot si želel)
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    # 5. PRETVORBA NAZAJ V NUMPY POLJA
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)