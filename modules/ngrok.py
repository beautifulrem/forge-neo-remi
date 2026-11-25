import os

import ngrok

# Connect to ngrok for ingress
def connect(token, port, options):
    account = None
    if token is None:
        token = 'None'
    else:
        if ':' in token:
            # token = authtoken:username:password
            token, username, password = token.split(':', 2)
            account = f"{username}:{password}"

    # For all options see: https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py
    if not options.get('authtoken_from_env'):
        options['authtoken'] = token
    if account:
        options['basic_auth'] = account
    if not options.get('session_metadata'):
        options['session_metadata'] = 'stable-diffusion-webui'


    try:
        public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
    except Exception as e:
        # the console is often logged: name the token by its end only, and keep secrets out of the reason
        secrets = [token, os.environ.get('NGROK_AUTHTOKEN')]
        auths = options.get('basic_auth') or []  # the token's user:password, or --ngrok-options'
        for auth in [auths] if isinstance(auths, str) else auths:
            if isinstance(auth, str):
                secrets += [auth, auth.split(':', 1)[-1]]
        reason = str(e)
        for secret in sorted((secret for secret in secrets if secret and len(secret) > 4), key=len, reverse=True):
            reason = reason.replace(secret, '***')
        shown = token if token == 'None' else f'...{token[-4:]}' if len(token) > 8 else '***'
        print(f'Invalid ngrok authtoken? ngrok connection aborted due to: {reason}\n'
              f'Your token: {shown}, get the right one on https://dashboard.ngrok.com/get-started/your-authtoken')
    else:
        print(f'ngrok connected to localhost:{port}! URL: {public_url}\n'
               'You can use this link after the launch is complete.')
