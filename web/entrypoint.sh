#!/bin/sh

echo "Waiting for postgres..."
echo $POSTGRES_HOST
echo $POSTGRES_PORT

# while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
#     sleep 0.1
# done

# echo "PostgreSQL started"

python3 manage.py makemigrations
python3 manage.py migrate

exec "$@"