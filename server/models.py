# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Agence(models.Model):
    wilaya = models.ForeignKey('Wilaya', models.DO_NOTHING, db_column='wilaya')
    name = models.TextField()
    longitude = models.FloatField(db_column='Longitude')  # Field name made lowercase.
    latitude = models.FloatField(db_column='Latitude')  # Field name made lowercase.
    id = models.UUIDField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'Agence'


class ApiProspect(models.Model):
    id = models.BigAutoField(primary_key=True)
    pos_name = models.TextField()
    longitude = models.FloatField()
    latitude = models.FloatField()
    status = models.CharField(max_length=10)
    person_name = models.TextField()
    phone_number = models.CharField(max_length=128)
    street_address = models.TextField(blank=True, null=True)
    pos_categorie = models.TextField()
    notes = models.TextField(blank=True, null=True)
    pos_photo = models.CharField(max_length=100, blank=True, null=True)
    commune = models.ForeignKey('Commune', models.DO_NOTHING, db_column='commune')

    class Meta:
        managed = False
        db_table = 'api_prospect'


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class AuthtokenToken(models.Model):
    key = models.CharField(primary_key=True, max_length=40)
    created = models.DateTimeField()
    user = models.OneToOneField(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'authtoken_token'


class Commune(models.Model):
    wilaya = models.ForeignKey('Wilaya', models.DO_NOTHING, db_column='wilaya')
    name = models.TextField()
    id = models.UUIDField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'commune'


class Conversation(models.Model):
    created_at = models.DateTimeField()
    member1 = models.ForeignKey('User', models.DO_NOTHING, db_column='member1', blank=True, null=True)
    member2 = models.ForeignKey('User', models.DO_NOTHING, db_column='member2', related_name='conversation_member2_set', blank=True, null=True)
    id = models.UUIDField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'conversation'


class Coordinates(models.Model):
    id = models.UUIDField(primary_key=True)
    created_at = models.DateTimeField()
    longitude = models.FloatField(blank=True, null=True)
    lattitude = models.FloatField(blank=True, null=True)
    user = models.ForeignKey('User', models.DO_NOTHING, db_column='user', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'coordinates'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Message(models.Model):
    created_at = models.DateTimeField()
    text = models.TextField(blank=True, null=True)
    sender = models.ForeignKey('User', models.DO_NOTHING, db_column='sender', blank=True, null=True)
    receiver = models.ForeignKey('User', models.DO_NOTHING, db_column='receiver', related_name='message_receiver_set', blank=True, null=True)
    conversation = models.ForeignKey(Conversation, models.DO_NOTHING, db_column='conversation', blank=True, null=True)
    id = models.UUIDField(primary_key=True)
    readed = models.BooleanField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'message'


class PointOfSale(models.Model):
    last_visit = models.DateTimeField(blank=True, null=True)
    longitude = models.FloatField()
    latitude = models.FloatField()
    commune = models.ForeignKey(Commune, models.DO_NOTHING, db_column='commune')
    zone = models.ForeignKey('Zone', models.DO_NOTHING, db_column='zone', blank=True, null=True)
    id = models.UUIDField(primary_key=True)
    status = models.SmallIntegerField()
    manager = models.ForeignKey('User', models.DO_NOTHING, db_column='manager', blank=True, null=True)
    created_at = models.DateTimeField()
    name = models.TextField()
    opening_time = models.TimeField(blank=True, null=True)
    closing_time = models.TimeField(blank=True, null=True)
    store_type = models.TextField(blank=True, null=True)
    contact = models.TextField(blank=True, null=True)
    type = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'point_of_sale'


class Prospect(models.Model):
    id = models.BigAutoField(primary_key=True)
    pos_name = models.TextField()
    longitude = models.FloatField()
    latitude = models.FloatField()
    status = models.CharField(max_length=10)
    phone_number = models.CharField(max_length=128)
    street_address = models.TextField()
    commune = models.ForeignKey(Commune, models.DO_NOTHING, db_column='commune')
    registre_de_commerce = models.TextField()
    numero_fiscal = models.TextField()

    class Meta:
        managed = False
        db_table = 'prospect'


class ProspectObjectives(models.Model):
    id = models.UUIDField(primary_key=True)
    created_at = models.DateTimeField()
    user = models.ForeignKey('User', models.DO_NOTHING, db_column='user')
    objective = models.IntegerField()
    deadline = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'prospect_objectives'


class TokenBlacklistBlacklistedtoken(models.Model):
    id = models.BigAutoField(primary_key=True)
    blacklisted_at = models.DateTimeField()
    token = models.OneToOneField('TokenBlacklistOutstandingtoken', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'token_blacklist_blacklistedtoken'


class TokenBlacklistOutstandingtoken(models.Model):
    id = models.BigAutoField(primary_key=True)
    token = models.TextField()
    created_at = models.DateTimeField(blank=True, null=True)
    expires_at = models.DateTimeField()
    user = models.ForeignKey(AuthUser, models.DO_NOTHING, blank=True, null=True)
    jti = models.CharField(unique=True, max_length=255)

    class Meta:
        managed = False
        db_table = 'token_blacklist_outstandingtoken'


class User(models.Model):
    created_at = models.DateTimeField(blank=True, null=True)
    role = models.TextField()
    wilaya = models.ForeignKey('Wilaya', models.DO_NOTHING, db_column='wilaya', blank=True, null=True)
    manager = models.ForeignKey('self', models.DO_NOTHING, db_column='manager', blank=True, null=True)
    phone = models.TextField()
    email = models.TextField()
    password = models.TextField()
    last_seen = models.DateTimeField(blank=True, null=True)
    id = models.UUIDField(primary_key=True)
    first_name = models.TextField(blank=True, null=True)
    last_name = models.TextField(blank=True, null=True)
    status = models.TextField(blank=True, null=True)
    profile_picture = models.TextField(blank=True, null=True)
    agence = models.ForeignKey(Agence, models.DO_NOTHING, db_column='Agence', blank=True, null=True)  # Field name made lowercase.
    deadline = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'user'


class Visit(models.Model):
    id = models.UUIDField(primary_key=True)
    visit_time = models.DateTimeField(blank=True, null=True)
    agent = models.ForeignKey(User, models.DO_NOTHING, db_column='agent', blank=True, null=True)
    pdv = models.ForeignKey(PointOfSale, models.DO_NOTHING, db_column='pdv', blank=True, null=True)
    duration = models.IntegerField(blank=True, null=True)
    status = models.TextField(blank=True, null=True)
    deadline = models.DateTimeField()
    order = models.SmallIntegerField()
    validated = models.SmallIntegerField()
    cancel_proof = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'visit'


class Wilaya(models.Model):
    name = models.TextField()
    id = models.UUIDField(primary_key=True)
    geojson = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'wilaya'


class Zone(models.Model):
    created_at = models.DateTimeField()
    commune = models.ForeignKey(Commune, models.DO_NOTHING, db_column='commune', blank=True, null=True)
    id = models.UUIDField(primary_key=True)
    manager = models.ForeignKey(User, models.DO_NOTHING, db_column='manager', blank=True, null=True)
    name = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'zone'
