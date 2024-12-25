

Illya Nokhrin 8:59 AM

Hi all, I was hoping someone might be able to help me with an issue I'm having trying to restore a postgresql do dump to a postgresql container. I've put the dump file into my backups directory and if I run docker compose exec postgres Is -1 /backups | get:

-rwxrwxrwx 1 root root

88087479 May

25 14:12 db_dump.sql

-rwxrwxrwx 1 root root 32371884 May 24 16:34 db_dump.sql.gz

But then if I run docker compose exec postgres restore db _dump.sql, I only get an error saying '':

No such file or directory. Any ideas what I've done wrong here? (edited)


> [!ai]+ AI
>
> It looks like you're encountering an issue where the PostgreSQL container cannot find the dump file you are trying to restore. Let's go through some steps to troubleshoot and resolve this issue.
> 1. **Verify File Location Inside the Container:**
> First, ensure that the dump file is indeed located at `/backups/db_dump.sql` inside the container. You can do this by listing the files in the `/backups` directory inside your PostgreSQL container:
> ```sh
> docker compose exec postgres ls -l /backups
> ```
> This should confirm that `db_dump.sql` exists at that location.
> 2. **Check Volume Mapping:**
> Ensure that your `docker-compose.yml` correctly maps a volume from your host machine to the container. It should look something like this:
> ```yaml
> version: '3'
> services:
> postgres:
> image: postgres:latest
> volumes:
> - ./path/to/your/local/backups:/backups
> environment:
> POSTGRES_DB: exampledb
> POSTGRES_USER: exampleuser
> POSTGRES_PASSWORD: examplepass
> ```
> 3. **Execute Restore Command Correctly:**
> The command you are using to restore might not be correct. Instead of running `docker compose exec postgres restore db_dump.sql`, use `psql` or `pg_restore` depending on whether your dump file is in plain SQL format or a custom format.
> For a plain SQL dump file (`db_dump.sql`):
> ```sh
> docker compose exec -T postgres psql -U <username> -d <database> -f /backups/db_dump.sql
> ```
> For a custom-format dump file (`db_dump.sql.gz`, assuming it's compressed with gzip):
> ```sh
> zcat path/to/local/db_dump.sql.gz | docker compose exec -T postgres pg_restore -U <username> -d <database>
> ```
> 4. **Check Permissions and Ownerships:**
> Ensure that your PostgreSQL container has appropriate permissions to access and read the dump files.
> 5. **Environment Variables and Credentials:**
> Ensure that you replace `<username>` and `<database>` with actual values for your PostgreSQL setup:
> ```sh
> docker compose exec -T postgres psql -U exampleuser -d exampledb -f /backups/db_dump.sql
> ```
> 6. **Docker Compose Version Compatibility:**
> Ensure you are using compatible versions of Docker Compose commands as syntax might slightly differ between versions.
> Here’s an example with specific values filled in:
> ```sh
> docker compose exec -T postgres psql -U myuser -d mydatabase -f /backups/db_dump.sql
> ```
> After following these steps, you should be able to restore your PostgreSQL dump file without encountering any "No such file or directory" errors. If problems persist, please share more details about error messages or logs for further assistance.


> [!failure]- Failure 
>   Error: Selection points outside of document
>  
