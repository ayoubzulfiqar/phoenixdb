//! Role-based access control.
//!
//! # Model
//!
//! A **principal** (user) holds a set of **roles**; each role grants a set of
//! [`Permission`]s. An operation is allowed when any of the principal's roles
//! grants the required permission. Denials are never cached and never inferred:
//! an unknown token is denied, an unknown role grants nothing.
//!
//! # Constant-time token comparison
//!
//! Tokens are looked up by comparing every candidate with
//! [`ct_eq`](crate::security::ct_eq), so the time taken does not depend on how
//! many leading bytes matched. A naive `HashMap` lookup would leak token
//! prefixes through timing; the linear scan here is deliberate and is why the
//! table is expected to hold tens, not millions, of principals.
//!
//! # Fail-closed
//!
//! Every path returns [`Error::InvalidArgument`] on denial rather than a
//! boolean, so a caller that forgets to check the result cannot accidentally
//! proceed — the `?` operator stops them.

use crate::error::{Error, Result};
use crate::security::ct_eq;
use std::collections::{BTreeSet, HashMap};

/// A capability that can be granted to a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Permission {
    /// Read a key.
    Read = 0,
    /// Create or overwrite a key.
    Write = 1,
    /// Remove a key.
    Delete = 2,
    /// Checkpoint, flush, compact, verify.
    Maintain = 3,
    /// Manage principals, roles and grants.
    Admin = 4,
}

impl Permission {
    /// Every permission, for granting a superuser role.
    #[must_use]
    pub fn all() -> [Permission; 5] {
        [
            Permission::Read,
            Permission::Write,
            Permission::Delete,
            Permission::Maintain,
            Permission::Admin,
        ]
    }

    /// Stable name used in audit records.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Delete => "delete",
            Permission::Maintain => "maintain",
            Permission::Admin => "admin",
        }
    }
}

/// A named set of permissions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Role {
    /// Role name, unique within a [`AccessControl`].
    pub name: String,
    /// Permissions this role grants.
    pub permissions: BTreeSet<Permission>,
}

impl Role {
    /// Creates a role granting `permissions`.
    #[must_use]
    pub fn new(name: impl Into<String>, permissions: impl IntoIterator<Item = Permission>) -> Self {
        Role {
            name: name.into(),
            permissions: permissions.into_iter().collect(),
        }
    }

    /// A role granting every permission.
    #[must_use]
    pub fn superuser(name: impl Into<String>) -> Self {
        Role::new(name, Permission::all())
    }

    /// A role granting read-only access.
    #[must_use]
    pub fn read_only(name: impl Into<String>) -> Self {
        Role::new(name, [Permission::Read])
    }

    /// True when this role grants `permission`.
    #[must_use]
    pub fn grants(&self, permission: Permission) -> bool {
        self.permissions.contains(&permission)
    }
}

/// An authenticated identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    /// Stable identifier recorded in the audit log.
    pub user_id: String,
    /// Roles held by this principal.
    pub roles: BTreeSet<String>,
}

/// The permission table: principals, roles, and the tokens that authenticate.
///
/// Tokens are stored as raw bytes and compared in constant time. In a real
/// deployment they should be high-entropy random values; this type does not
/// generate them because entropy sourcing belongs to the embedder.
#[derive(Debug, Default)]
pub struct AccessControl {
    roles: HashMap<String, Role>,
    /// `(token, principal)` pairs, scanned in constant time.
    principals: Vec<(Vec<u8>, Principal)>,
    /// When true, an empty token maps to a built-in full-access principal.
    ///
    /// This preserves backwards compatibility for embedded single-user
    /// deployments that never opt into RBAC.
    open_by_default: bool,
}

impl AccessControl {
    /// Creates an empty, **fail-closed** table: every request is denied until
    /// a principal is registered.
    #[must_use]
    pub fn new() -> Self {
        AccessControl {
            roles: HashMap::new(),
            principals: Vec::new(),
            open_by_default: false,
        }
    }

    /// Creates a table that allows everything when no principal is registered.
    ///
    /// This is the mode an existing embedded application gets by default, so
    /// enabling the security module does not silently break it. As soon as one
    /// principal is added, the table becomes fail-closed.
    #[must_use]
    pub fn open() -> Self {
        AccessControl {
            roles: HashMap::new(),
            principals: Vec::new(),
            open_by_default: true,
        }
    }

    /// True when no principals are registered and the table is in open mode.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open_by_default && self.principals.is_empty()
    }

    /// Registers or replaces a role.
    pub fn define_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Looks up a role by name.
    #[must_use]
    pub fn role(&self, name: &str) -> Option<&Role> {
        self.roles.get(name)
    }

    /// Registers a principal authenticated by `token`.
    ///
    /// Returns [`Error::InvalidArgument`] for an empty token, a duplicate
    /// token, or a role that has not been defined — failing loudly beats
    /// registering a principal whose roles silently grant nothing.
    pub fn add_principal(
        &mut self,
        token: impl Into<Vec<u8>>,
        user_id: impl Into<String>,
        roles: impl IntoIterator<Item = String>,
    ) -> Result<()> {
        let token = token.into();
        if token.is_empty() {
            return Err(Error::invalid("authentication token must not be empty"));
        }
        if self.principals.iter().any(|(t, _)| ct_eq(t, &token)) {
            return Err(Error::invalid("token is already registered"));
        }
        let roles: BTreeSet<String> = roles.into_iter().collect();
        for role in &roles {
            if !self.roles.contains_key(role) {
                return Err(Error::invalid(format!("role {role:?} is not defined")));
            }
        }
        self.principals.push((
            token,
            Principal {
                user_id: user_id.into(),
                roles,
            },
        ));
        Ok(())
    }

    /// Removes the principal authenticated by `token`.
    ///
    /// Returns `true` when a principal was removed.
    pub fn remove_principal(&mut self, token: &[u8]) -> bool {
        // Scan every entry so removal timing does not reveal position.
        let mut found = None;
        for (i, (t, _)) in self.principals.iter().enumerate() {
            if ct_eq(t, token) {
                found = Some(i);
            }
        }
        match found {
            Some(i) => {
                self.principals.remove(i);
                true
            }
            None => false,
        }
    }

    /// Number of registered principals.
    #[must_use]
    pub fn principal_count(&self) -> usize {
        self.principals.len()
    }

    /// Resolves `token` to its principal in constant time.
    ///
    /// Every entry is compared even after a match, so lookup time does not
    /// depend on where in the table the token sits.
    #[must_use]
    pub fn authenticate(&self, token: &[u8]) -> Option<&Principal> {
        let mut found: Option<&Principal> = None;
        for (candidate, principal) in &self.principals {
            if ct_eq(candidate, token) {
                found = Some(principal);
            }
        }
        found
    }

    /// Checks that `token` grants `permission`.
    ///
    /// Returns the authenticated [`Principal`] on success so the caller can
    /// record `user_id` in the audit log without a second lookup.
    pub fn authorize(&self, token: &[u8], permission: Permission) -> Result<&Principal> {
        if self.is_open() {
            return Err(Error::invalid(
                "access control is in open mode; call authorize_open instead",
            ));
        }
        let principal = self
            .authenticate(token)
            .ok_or_else(|| Error::invalid("authentication failed: unknown token"))?;
        for role_name in &principal.roles {
            if let Some(role) = self.roles.get(role_name)
                && role.grants(permission)
            {
                return Ok(principal);
            }
        }
        Err(Error::invalid(format!(
            "principal {:?} lacks the {} permission",
            principal.user_id,
            permission.name()
        )))
    }

    /// Like [`AccessControl::authorize`], but permits everything while the
    /// table is in open mode with no principals registered.
    ///
    /// Returns `None` for the open case (no identity to attribute), `Some` once
    /// RBAC is actually configured.
    pub fn authorize_open(
        &self,
        token: &[u8],
        permission: Permission,
    ) -> Result<Option<&Principal>> {
        if self.is_open() {
            return Ok(None);
        }
        self.authorize(token, permission).map(Some)
    }

    /// Every permission `token` holds, for diagnostics.
    #[must_use]
    pub fn effective_permissions(&self, token: &[u8]) -> BTreeSet<Permission> {
        let mut out = BTreeSet::new();
        if let Some(principal) = self.authenticate(token) {
            for role_name in &principal.roles {
                if let Some(role) = self.roles.get(role_name) {
                    out.extend(role.permissions.iter().copied());
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured() -> AccessControl {
        let mut ac = AccessControl::new();
        ac.define_role(Role::superuser("admin"));
        ac.define_role(Role::read_only("reader"));
        ac.define_role(Role::new("writer", [Permission::Read, Permission::Write]));
        ac.add_principal(b"admin-token".to_vec(), "root", ["admin".to_string()])
            .unwrap();
        ac.add_principal(b"reader-token".to_vec(), "alice", ["reader".to_string()])
            .unwrap();
        ac.add_principal(b"writer-token".to_vec(), "bob", ["writer".to_string()])
            .unwrap();
        ac
    }

    #[test]
    fn superuser_holds_every_permission() {
        let ac = configured();
        for p in Permission::all() {
            assert!(
                ac.authorize(b"admin-token", p).is_ok(),
                "admin denied {p:?}"
            );
        }
    }

    #[test]
    fn reader_cannot_write_or_delete() {
        let ac = configured();
        assert!(ac.authorize(b"reader-token", Permission::Read).is_ok());
        assert!(ac.authorize(b"reader-token", Permission::Write).is_err());
        assert!(ac.authorize(b"reader-token", Permission::Delete).is_err());
        assert!(ac.authorize(b"reader-token", Permission::Admin).is_err());
    }

    #[test]
    fn writer_can_write_but_not_administer() {
        let ac = configured();
        assert!(ac.authorize(b"writer-token", Permission::Write).is_ok());
        assert!(ac.authorize(b"writer-token", Permission::Read).is_ok());
        assert!(ac.authorize(b"writer-token", Permission::Delete).is_err());
        assert!(ac.authorize(b"writer-token", Permission::Admin).is_err());
    }

    #[test]
    fn unknown_token_is_denied() {
        let ac = configured();
        assert!(ac.authorize(b"forged", Permission::Read).is_err());
        assert!(ac.authorize(b"", Permission::Read).is_err());
        // A token that is a prefix of a real one must not authenticate.
        assert!(ac.authorize(b"admin-toke", Permission::Read).is_err());
        assert!(ac.authorize(b"admin-tokenX", Permission::Read).is_err());
    }

    #[test]
    fn authorize_returns_the_principal_for_auditing() {
        let ac = configured();
        let p = ac.authorize(b"writer-token", Permission::Write).unwrap();
        assert_eq!(p.user_id, "bob");
    }

    #[test]
    fn empty_table_denies_everything_when_fail_closed() {
        let ac = AccessControl::new();
        assert!(!ac.is_open());
        assert!(ac.authorize(b"anything", Permission::Read).is_err());
    }

    #[test]
    fn open_mode_permits_until_a_principal_is_registered() {
        let mut ac = AccessControl::open();
        assert!(ac.is_open());
        // No identity, but allowed: preserves single-user embedded behaviour.
        assert_eq!(ac.authorize_open(b"", Permission::Write).unwrap(), None);

        ac.define_role(Role::read_only("reader"));
        ac.add_principal(b"t".to_vec(), "alice", ["reader".to_string()])
            .unwrap();
        assert!(!ac.is_open(), "registering a principal closes the door");
        assert!(ac.authorize_open(b"", Permission::Write).is_err());
        assert!(ac.authorize_open(b"t", Permission::Read).unwrap().is_some());
    }

    #[test]
    fn duplicate_and_empty_tokens_are_rejected() {
        let mut ac = AccessControl::new();
        ac.define_role(Role::read_only("reader"));
        ac.add_principal(b"tok".to_vec(), "a", ["reader".to_string()])
            .unwrap();
        assert!(
            ac.add_principal(b"tok".to_vec(), "b", ["reader".to_string()])
                .is_err(),
            "duplicate token must be rejected"
        );
        assert!(
            ac.add_principal(Vec::new(), "c", ["reader".to_string()])
                .is_err(),
            "empty token must be rejected"
        );
    }

    #[test]
    fn undefined_role_is_rejected_at_registration() {
        let mut ac = AccessControl::new();
        let err = ac.add_principal(b"t".to_vec(), "a", ["ghost".to_string()]);
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
        assert_eq!(ac.principal_count(), 0);
    }

    #[test]
    fn multiple_roles_union_their_permissions() {
        let mut ac = AccessControl::new();
        ac.define_role(Role::read_only("reader"));
        ac.define_role(Role::new("deleter", [Permission::Delete]));
        ac.add_principal(
            b"t".to_vec(),
            "multi",
            ["reader".to_string(), "deleter".to_string()],
        )
        .unwrap();
        assert!(ac.authorize(b"t", Permission::Read).is_ok());
        assert!(ac.authorize(b"t", Permission::Delete).is_ok());
        assert!(ac.authorize(b"t", Permission::Write).is_err());

        let effective = ac.effective_permissions(b"t");
        assert!(effective.contains(&Permission::Read));
        assert!(effective.contains(&Permission::Delete));
        assert!(!effective.contains(&Permission::Write));
    }

    #[test]
    fn removing_a_principal_revokes_access() {
        let mut ac = configured();
        assert!(ac.authorize(b"reader-token", Permission::Read).is_ok());
        assert!(ac.remove_principal(b"reader-token"));
        assert!(ac.authorize(b"reader-token", Permission::Read).is_err());
        assert!(
            !ac.remove_principal(b"reader-token"),
            "second removal is a no-op"
        );
    }

    #[test]
    fn redefining_a_role_changes_effective_access() {
        let mut ac = configured();
        assert!(ac.authorize(b"reader-token", Permission::Write).is_err());
        // Widen the reader role; the existing principal picks it up.
        ac.define_role(Role::new("reader", [Permission::Read, Permission::Write]));
        assert!(ac.authorize(b"reader-token", Permission::Write).is_ok());
    }

    #[test]
    fn permission_names_are_stable_for_audit_records() {
        assert_eq!(Permission::Read.name(), "read");
        assert_eq!(Permission::Write.name(), "write");
        assert_eq!(Permission::Delete.name(), "delete");
        assert_eq!(Permission::Maintain.name(), "maintain");
        assert_eq!(Permission::Admin.name(), "admin");
    }
}
