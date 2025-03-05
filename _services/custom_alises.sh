# Megaton-roto shortcuts
alias backlog='sudo journalctl -u megaton-roto.service -f'
alias frontlog='sudo journalctl -u megaton-roto-frontend.service -f'
alias backend='cd /home/ec2-user/megaton-roto/backend && ./start.sh'

# View frontend development service logs
alias backdevlog='sudo journalctl -u megaton-roto-dev.service -f'

# View backend development service logs
alias frontdevlog='sudo journalctl -u megaton-roto-frontend-dev.service -f'

# View production frontend logs
alias frontlog='sudo journalctl -u megaton-roto-frontend.service -f'

# View Nginx logs
alias nginxlog='sudo tail -f /var/log/nginx/error.log'

# sudo cp /home/ec2-user/megaton-roto-dev/_services/custom_alises.sh /etc/profile.d/custom-aliases.sh
# source /etc/profile.d/custom-aliases.sh
